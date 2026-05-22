# src/evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def get_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "accuracy": accuracy,
        "classification_report": report
    }


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(4, 3))
    sns.set_theme(style="white")

    sns.heatmap(
        df_cm,
        annot=True,
        fmt="d",
        cmap="coolwarm",
        cbar=True,
        linewidths=1.5,
        annot_kws={
            "size": 16,
            "weight": "bold"
        }
    )

    plt.title("Confusion Matrix", fontsize=18, pad=20)
    plt.ylabel("Actual Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.show()


def show_classification_report(y_true, y_pred):
    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    df_report = pd.DataFrame(report_dict).transpose()

    return df_report.style.background_gradient(
        cmap="RdYlGn",
        subset=pd.IndexSlice[
            df_report.index[:-3],
            ["precision", "recall", "f1-score"]
        ]
    )


def get_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    return importance_df