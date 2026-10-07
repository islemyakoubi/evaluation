"""Reproduce rolling validation, both prediction files, and the supplied chart.

Run `python fit_submission.py` from the project directory. All evaluation fits
use earlier labeled rows only; final-validation labels are never available.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path

from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from freight_prep import (
    ROOT, TARGET, CATEGORICAL_FEATURES, CYCLIC_FEATURES, CYCLIC_CORE_FEATURES,
    RICH_FEATURES, DECEMBER_COLUMNS, add_candidate_features, audit_data,
    load_raw_data, validate_sources,
)
from score import save_december_chart, validate_december, validate_predictions

TREE_WEIGHT = 0.5
THREADS = 4
MAX_ITERATIONS = 600
EARLY_STOPPING_ROUNDS = 60
ROBUST_CATEGORIES = ["pickup", "delivery", "equipment"]
ROBUST_NUMERIC = ["inverse_distance", "weight", "market_index", "year_sin", "year_cos", "day_of_week"]
FOLDS = [
    ("May-June", "2025-05-01", "2025-07-01"),
    ("July-August", "2025-07-01", "2025-09-01"),
    ("September-October", "2025-09-01", "2025-11-01"),
]
MODEL_NAMES = [
    "Training median", "Robust rich regression", "Seasonal rich CatBoost",
    "Selected rich ensemble", "CatBoost with quote signal",
    "Robust core regression", "Seasonal core CatBoost RPM", "Selected core ensemble",
]


def regression_metrics(y, predictions) -> dict:
    y, predictions = np.asarray(y, dtype=float), np.asarray(predictions, dtype=float)
    errors = predictions - y
    return {
        "rows": len(y), "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(mean_squared_error(y, predictions) ** 0.5),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "p90_absolute_error": float(np.quantile(np.abs(errors), .90)),
        "mean_error": float(np.mean(errors)),
    }


class RobustRateModel:
    """Distance interactions with robust dollar loss and pooled unseen cities.

    Numeric statistics, categorical vocabulary, and population fallback weights
    are fitted only on the supplied training partition. Multiplying the design
    by distance lets equipment and city effects represent per-mile premiums.
    """

    def __init__(self, core: bool = False):
        self.numeric = [f for f in ROBUST_NUMERIC if not (core and f == "market_index")]
        self.preprocessor = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), self.numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ROBUST_CATEGORIES),
        ])
        self.regressor = HuberRegressor(
            epsilon=1.35, alpha=.01, max_iter=2000, tol=1e-6, fit_intercept=False,
        )
        self.population_weights = []

    def fit(self, train: pd.DataFrame):
        matrix = sparse.csr_matrix(self.preprocessor.fit_transform(train))
        encoder = self.preprocessor.named_transformers_["categorical"]
        self.population_weights = [
            train[column].value_counts(normalize=True).reindex(values).to_numpy()
            for column, values in zip(ROBUST_CATEGORIES, encoder.categories_)
        ]
        design = matrix.multiply(train["distance"].to_numpy()[:, None] / 1000)
        self.regressor.fit(design, train[TARGET])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = sparse.csr_matrix(self.preprocessor.transform(frame)).tolil()
        encoder = self.preprocessor.named_transformers_["categorical"]
        start = len(self.numeric)
        for column, values, weights in zip(ROBUST_CATEGORIES, encoder.categories_, self.population_weights):
            unseen = np.flatnonzero(~frame[column].isin(values).to_numpy())
            if len(unseen):
                matrix[np.ix_(unseen, np.arange(start, start + len(values)))] = np.tile(weights, (len(unseen), 1))
            start += len(values)
        design = matrix.tocsr().multiply(frame["distance"].to_numpy()[:, None] / 1000)
        return self.regressor.predict(design)


@dataclass
class FittedTree:
    model: CatBoostRegressor
    features: list[str]
    normalized_target: bool
    iterations: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        result = self.model.predict(frame[self.features])
        return result * frame["distance"].to_numpy() if self.normalized_target else result


def fit_tree(train: pd.DataFrame, core: bool = False, legacy: bool = False) -> FittedTree:
    features = RICH_FEATURES if legacy else (CYCLIC_CORE_FEATURES if core else CYCLIC_FEATURES)
    normalized = core and not legacy
    # Early stopping uses the last month WITHIN training, never the outer fold.
    cutoff = train["date"].max().to_period("M").to_timestamp()
    inner_train, inner_validation = train[train["date"] < cutoff], train[train["date"] >= cutoff]
    if inner_train.empty or inner_validation.empty:
        raise ValueError("At least two training months are required for early stopping.")
    params = {
        "depth": 4 if normalized else 6, "learning_rate": .05,
        "loss_function": "MAE", "eval_metric": "MAE", "random_seed": 42,
        "thread_count": THREADS, "verbose": False, "allow_writing_files": False,
        "counter_calc_method": "SkipTest",
    }

    def pool(frame):
        y = frame[TARGET] / frame["distance"] if normalized else frame[TARGET]
        # Weighted RPM MAE is dollar MAE up to a positive constant; use the
        # same weights for both learning and early stopping.
        return Pool(frame[features], y, cat_features=CATEGORICAL_FEATURES,
                    weight=frame["distance"] if normalized else None)

    probe = CatBoostRegressor(iterations=MAX_ITERATIONS, **params)
    probe.fit(pool(inner_train), eval_set=pool(inner_validation),
              early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True)
    iterations = int(probe.tree_count_)
    final = CatBoostRegressor(iterations=iterations, **params)
    final.fit(pool(train))
    return FittedTree(final, features, normalized, iterations)


def ensemble_predictions(tree: FittedTree, robust: RobustRateModel, frame: pd.DataFrame) -> np.ndarray:
    result = TREE_WEIGHT * tree.predict(frame) + (1 - TREE_WEIGHT) * robust.predict(frame)
    if not np.isfinite(result).all():
        raise ValueError("Model produced a non-finite prediction.")
    return np.maximum(result, 1.0)


def evaluate_rolling(development: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    records, prediction_frames, folds = [], [], []
    for label, start, end in FOLDS:
        train = development[development["date"] < start]
        test = development[(development["date"] >= start) & (development["date"] < end)]
        if train.empty or test.empty or train["date"].max() >= test["date"].min():
            raise ValueError(f"Invalid forward split: {label}")
        inner_cutoff = train["date"].max().to_period("M").to_timestamp()
        folds.append({
            "fold": label, "training_start": str(train["date"].min().date()),
            "training_end": str(train["date"].max().date()),
            "test_start": str(test["date"].min().date()), "test_end": str(test["date"].max().date()),
            "training_rows": len(train), "test_rows": len(test),
            "inner_training_rows": int((train["date"] < inner_cutoff).sum()),
            "early_stopping_rows": int((train["date"] >= inner_cutoff).sum()),
        })
        rich_robust, core_robust = RobustRateModel().fit(train), RobustRateModel(core=True).fit(train)
        rich_tree, core_tree, legacy_tree = fit_tree(train), fit_tree(train, core=True), fit_tree(train, legacy=True)
        values = {
            "Training median": np.full(len(test), train[TARGET].median()),
            "Robust rich regression": rich_robust.predict(test),
            "Seasonal rich CatBoost": rich_tree.predict(test),
            "Selected rich ensemble": ensemble_predictions(rich_tree, rich_robust, test),
            "CatBoost with quote signal": legacy_tree.predict(test),
            "Robust core regression": core_robust.predict(test),
            "Seasonal core CatBoost RPM": core_tree.predict(test),
            "Selected core ensemble": ensemble_predictions(core_tree, core_robust, test),
        }
        iterations = {"rich": rich_tree.iterations, "core": core_tree.iterations, "quote": legacy_tree.iterations}
        output = test[["load_id", "date", "equipment", "distance", TARGET]].copy()
        output["fold"] = label
        for name, predictions in values.items():
            predictions = np.maximum(predictions, 1.0)
            records.append({"fold": label, "model": name, **regression_metrics(test[TARGET], predictions)})
            output[name] = predictions
        folds[-1]["tree_iterations"] = iterations
        prediction_frames.append(output)
        print(f"{label}: rich MAE ${records[-5]['mae']:.2f}; core MAE ${records[-1]['mae']:.2f}", flush=True)
    pooled = pd.concat(prediction_frames, ignore_index=True)
    overall = [{"model": name, **regression_metrics(pooled[TARGET], pooled[name])} for name in MODEL_NAMES]
    selected = "Selected rich ensemble"
    segments = {
        "equipment": [{"equipment": key, **regression_metrics(group[TARGET], group[selected])}
                      for key, group in pooled.groupby("equipment")],
        "month": [{"month": key, **regression_metrics(group[TARGET], group[selected])}
                  for key, group in pooled.groupby(pooled["date"].dt.strftime("%Y-%m"))],
    }
    absolute_errors = (pooled[selected] - pooled[TARGET]).abs()
    threshold = float(absolute_errors.quantile(.99))
    segments["error_tail"] = {
        "p99_absolute_error": threshold,
        "top_one_percent_squared_error_share": float((absolute_errors[absolute_errors >= threshold] ** 2).sum() / (absolute_errors ** 2).sum()),
    }
    return {"score_role": "Internal rolling model-selection validation; no untouched labeled test set is claimed.",
            "folds": folds, "fold_metrics": records, "pooled_metrics": overall, "segments": segments}, pooled


def evaluate_unseen_cities(development: pd.DataFrame) -> dict:
    held_cities = ["Atlanta", "Dallas", "Denver", "Houston", "Miami", "Portland", "Seattle", "Tampa"]
    train = development[development["date"] < "2025-07-01"]
    train = train[~train["pickup"].isin(held_cities) & ~train["delivery"].isin(held_cities)]
    test = development[(development["date"] >= "2025-07-01") & (development["date"] < "2025-09-01")]
    test = test[test["pickup"].isin(held_cities) | test["delivery"].isin(held_cities)]
    rich_tree, rich_robust = fit_tree(train), RobustRateModel().fit(train)
    core_tree, core_robust = fit_tree(train, core=True), RobustRateModel(core=True).fit(train)
    return {
        "purpose": "Additional internal stress test, not an independent final score.",
        "held_cities": held_cities, "training_rows": len(train), "test_rows": len(test),
        "rich": regression_metrics(test[TARGET], ensemble_predictions(rich_tree, rich_robust, test)),
        "core": regression_metrics(test[TARGET], ensemble_predictions(core_tree, core_robust, test)),
    }


def predict_submission(development, validation, december, template, output_dir: Path) -> dict:
    rich_tree, rich_robust = fit_tree(development), RobustRateModel().fit(development)
    core_tree, core_robust = fit_tree(development, core=True), RobustRateModel(core=True).fit(development)
    rates = ensemble_predictions(rich_tree, rich_robust, add_candidate_features(validation))
    by_id = pd.Series(rates, index=validation["load_id"])
    predictions = template[["load_id"]].copy()
    predictions["predicted_rate"] = predictions["load_id"].map(by_id)
    validate_predictions(predictions)
    december_output = december[DECEMBER_COLUMNS].copy()
    december_output["predicted_rate"] = ensemble_predictions(core_tree, core_robust, add_candidate_features(december))
    december_checked = validate_december(december_output)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False, float_format="%.6f")
    december_output.to_csv(output_dir / "december_predictions.csv", index=False, float_format="%.6f")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Use the supplied scorer's plotting function without changing its design.
    submitted_december = validate_december(pd.read_csv(output_dir / "december_predictions.csv"))
    save_december_chart(submitted_december, results_dir / "candidate_december.png")
    return {
        "training_rows": len(development), "rich_iterations": rich_tree.iterations,
        "core_iterations": core_tree.iterations,
        "validation_rows": len(predictions), "december_rows": len(december_output),
        "validation_prediction_min": float(rates.min()), "validation_prediction_median": float(np.median(rates)),
        "validation_prediction_max": float(rates.max()),
        "december_prediction_min": float(december_output["predicted_rate"].min()),
        "december_prediction_max": float(december_output["predicted_rate"].max()),
        "id_order": "Exact order from the supplied prediction template.",
        "rich_features": rich_tree.features, "core_features": core_tree.features,
        "tree_importance": dict(zip(rich_tree.features, map(float, rich_tree.model.feature_importances_))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--skip-evaluation", action="store_true", help="Only refit and regenerate predictions and the chart.")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_train, validation, december, template = load_raw_data(args.data_dir.resolve())
    validate_sources(raw_train, validation, template)
    development = add_candidate_features(raw_train).sort_values("date", kind="stable").reset_index(drop=True)
    summary = {
        "audit": audit_data(raw_train, validation),
        "method": {
            "selection": "Three expanding-window, two-month forward validation periods.",
            "ensemble_tree_weight": TREE_WEIGHT,
            "tree_loss": "MAE", "tree_learning_rate": .05, "random_seed": 42,
            "rich_depth": 6, "core_depth": 4,
            "maximum_iterations": MAX_ITERATIONS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "early_stopping": "Last calendar month within each training partition, followed by refitting on that entire partition.",
            "core_target": "Rate per mile, with distance weights for both fitting and early stopping.",
            "robust": {"epsilon": 1.35, "alpha": .01, "max_iter": 2000, "tol": 1e-6},
            "excluded_features": ["load_id", "posted_rate as a predictor", "quote_signal in selected models"],
        },
        "environment": {package: importlib.metadata.version(package) for package in
                        ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "matplotlib")},
        "input_sha256": {filename: hashlib.sha256((args.data_dir / filename).read_bytes()).hexdigest()
                         for filename in ("train-test.csv", "validation.csv", "december-chart-inputs.csv", "validation-predictions-template.csv")},
    }
    if not args.skip_evaluation:
        summary["validation"], _ = evaluate_rolling(development)
        summary["unseen_city_stress"] = evaluate_unseen_cities(development)
    summary["final_fit"] = predict_submission(development, validation, december, template, output_dir)
    results_dir = output_dir / "results"
    summary_path = results_dir / ("prediction_run.json" if args.skip_evaluation else "validation_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Validated 12,000 load predictions and 31 fixed December predictions.", flush=True)
    print(f"Outputs saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
