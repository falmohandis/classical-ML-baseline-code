# src/experiment.py

import os

from sklearn.model_selection import train_test_split

from src.models import build_decision_tree, build_random_forest
from src.evaluation import (
    get_classification_metrics,
    get_feature_importance
)


def run_tree_experiment(
    df,
    target_col,
    feature_names=None,
    test_size=0.2,
    random_state=42,
    stratify=True,
    results_dir="results"
):
    os.makedirs(results_dir, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if feature_names is None:
        feature_names = X.columns.tolist()

    stratify_values = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values
    )

    models = {
        "Decision Tree": build_decision_tree(random_state=random_state),
        "Random Forest": build_random_forest(random_state=random_state)
    }

    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = get_classification_metrics(y_test, predictions)
        importance_df = get_feature_importance(model, feature_names)

        results[model_name] = {
            "model": model,
            "predictions": predictions,
            "accuracy": metrics["accuracy"],
            "classification_report": metrics["classification_report"],
            "feature_importance": importance_df,
            "X_test": X_test,
            "y_test": y_test
        }

        safe_name = model_name.lower().replace(" ", "_")
        importance_df.to_csv(
            f"{results_dir}/{safe_name}_feature_importance.csv",
            index=False
        )

    with open(f"{results_dir}/model_summary.txt", "w") as f:
        f.write("Classical ML Baseline\n")
        f.write("Models compared: Decision Tree and Random Forest\n\n")

        for model_name, result in results.items():
            f.write(f"{model_name} Accuracy: {result['accuracy']:.4f}\n")

    return results