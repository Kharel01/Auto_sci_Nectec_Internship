import json
import random
import time
from itertools import product
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "nn-digits-autotune"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(CODE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def make_model(hidden_dim, learning_rate, batch_size, alpha, max_iter, seed):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(hidden_dim,),
                    learning_rate_init=learning_rate,
                    batch_size=batch_size,
                    alpha=alpha,
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def run_seed(config, seed):
    data = load_digits()
    X = data.data
    y = data.target

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=seed,
        stratify=y_train_val,
    )

    max_iter = config["experiment"]["max_iter"]

    baseline_cfg = config["baseline"]
    baseline_model = make_model(
        baseline_cfg["hidden_dim"],
        baseline_cfg["learning_rate"],
        baseline_cfg["batch_size"],
        baseline_cfg["alpha"],
        max_iter,
        seed,
    )
    baseline_model.fit(X_train, y_train)

    baseline_val = accuracy_score(y_val, baseline_model.predict(X_val))
    baseline_test = accuracy_score(y_test, baseline_model.predict(X_test))

    space = config["search_space"]
    candidates = list(
        product(
            space["hidden_dim"],
            space["learning_rate"],
            space["batch_size"],
            space["alpha"],
        )
    )

    random.Random(seed).shuffle(candidates)
    candidates = candidates[: config["experiment"]["max_trials_per_seed"]]

    trials = []
    best = None

    for trial_id, (hidden_dim, learning_rate, batch_size, alpha) in enumerate(candidates, 1):
        start = time.time()

        model = make_model(
            hidden_dim,
            learning_rate,
            batch_size,
            alpha,
            max_iter,
            seed,
        )
        model.fit(X_train, y_train)

        val_acc = accuracy_score(y_val, model.predict(X_val))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        row = {
            "seed": seed,
            "trial_id": trial_id,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "alpha": alpha,
            "validation_accuracy": val_acc,
            "test_accuracy": test_acc,
            "training_time_seconds": time.time() - start,
        }

        trials.append(row)

        if best is None or row["validation_accuracy"] > best["validation_accuracy"]:
            best = row

    seed_result = {
        "seed": seed,
        "baseline": {
            **baseline_cfg,
            "validation_accuracy": baseline_val,
            "test_accuracy": baseline_test,
        },
        "best_tuned_model": best,
        "improvement_over_baseline": best["test_accuracy"] - baseline_test,
    }

    with open(RESULTS_DIR / f"seed_{seed}.json", "w") as f:
        json.dump(seed_result, f, indent=2)

    return trials, seed_result


def main():
    config = load_config()
    all_trials = []
    all_results = []

    for seed in config["experiment"]["seeds"]:
        print(f"Running seed {seed}...")
        trials, result = run_seed(config, seed)
        all_trials.extend(trials)
        all_results.append(result)

    pd.DataFrame(all_trials).to_csv(RESULTS_DIR / "trials.csv", index=False)

    baseline_mean = sum(r["baseline"]["test_accuracy"] for r in all_results) / len(all_results)
    tuned_mean = sum(r["best_tuned_model"]["test_accuracy"] for r in all_results) / len(all_results)

    best_result = max(all_results, key=lambda r: r["best_tuned_model"]["validation_accuracy"])

    best_config = {
        "best_seed": best_result["seed"],
        "best_config": {
            "hidden_dim": best_result["best_tuned_model"]["hidden_dim"],
            "learning_rate": best_result["best_tuned_model"]["learning_rate"],
            "batch_size": best_result["best_tuned_model"]["batch_size"],
            "alpha": best_result["best_tuned_model"]["alpha"],
        },
        "validation_accuracy": best_result["best_tuned_model"]["validation_accuracy"],
        "test_accuracy": best_result["best_tuned_model"]["test_accuracy"],
    }

    with open(RESULTS_DIR / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    summary = f"""# AutoSci Neural Network Tuning Demo

## Baseline mean test accuracy

{baseline_mean:.4f}

## Tuned mean test accuracy

{tuned_mean:.4f}

## Mean improvement

{tuned_mean - baseline_mean:.4f}

## Best configuration

{json.dumps(best_config, indent=2)}

## What this demonstrates

This is a complete AutoSci-style usage example. The project contains a research note, an idea page, an experiment plan, runnable neural-network training code, automatic hyperparameter tuning, saved results, and a final summary.
"""

    with open(RESULTS_DIR / "summary.md", "w") as f:
        f.write(summary)

    print(summary)
    print("Saved results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
