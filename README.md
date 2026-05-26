# Classical Machine Learning Baseline for Patient Risk Prediction

This is part 1 of a series on the various github projects I am developing to create pre-made "plug and play" machine & deep learning codes. My style of working with any kind of AI model is to start with a highly generalizable structure, and to then add unique features to the codes to match the exact problem at hand. Doing this is my way to ensure that the fundamentals of how to build these algorithms are respected while also adding problem-specific elements to the solution.


## Problem Statement

This project is centered on trying to predict disease risk from tabular patient data.

In the first sections of the code, I will synthetically generate patient-style tabular data. This synthetic dataset will- in a later iteration of the model- be replaced with Kaggle data to demonstrate both application and model tuning.

## Key Components of the Code:

- Tabular data handling
- Train/test splitting
- Baseline modeling
- Decision Tree classification
- Random Forest classification
- Classification metrics
- Feature importance interpretation

## Expceted results

In this study, we will compare Decision Tree and Random Forest classifiers on synthetic patient-style tabular data. Due to the increased complexity of random forest models by decreasing variance while maintaining bias,bringing in improved accuracy and providing interpretable feature importance.

## Data Sources for Decision Tree and Random Forest Testing

The following datasets will be used to test Decision Tree (DT) and Random Forest (RF) models across binary classification, multiclass classification, imbalanced classification, and regression tasks.

| Dataset | Task | Why it’s useful for DT/RF testing |
|---|---:|---|
| [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) | Binary classification | Classic starter dataset with mixed numeric and categorical features, missing values, and a clear survival prediction target. |
| [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) | Binary classification | Similar to Titanic, but more modern and feature-engineering-heavy. Useful for testing preprocessing pipelines and categorical handling. |
| [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) | Regression | Strong regression benchmark with many explanatory variables, categorical features, missing values, and feature engineering opportunities. |
| [Forest Cover Type Prediction](https://www.kaggle.com/c/forest-cover-type-prediction) | Multiclass classification | Useful for testing multiclass classification. The goal is to predict the predominant forest cover type from cartographic variables. |
| [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Imbalanced binary classification | Good test for class imbalance, with 492 fraud cases out of 284,807 transactions. Accuracy can be misleading, so F1, ROC-AUC, and PR-AUC should be considered. |
| [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) | Binary classification | Practical business-style dataset that is useful for categorical preprocessing, feature importance, and explainability. |
| [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) | Binary classification | Medical and demographic feature set useful for testing probability outputs, threshold tuning, and feature importance. |
| [Heart Disease Prediction](https://www.kaggle.com/datasets/jamiabdullahsummer22/heart-disease-prediction) | Binary classification | Small-to-medium medical classification dataset. Useful for comparing overfitting behavior between a single decision tree and a random forest. |
| [Student Performance Prediction](https://www.kaggle.com/datasets/amrmaree/student-performance-prediction) | Regression or classification | Flexible target framing depending on whether final exam score is treated as a continuous value or grouped into categories. |
| [Weather Forecast Dataset](https://www.kaggle.com/datasets/zeeshier/weather-forecast-dataset) | Classification | Beginner-friendly weather classification dataset with roughly 2,500 observations. Useful for quick model validation. |
| [Campus Placement Prediction](https://www.kaggle.com/datasets/meruvulikith/campus-selection-classification-dataset) | Binary classification | Small structured dataset that is useful for smoke testing pipelines and categorical encoding. |
| [Rock or Mine Classification](https://www.kaggle.com/datasets/vijayaadithyanvg/rock-or-mine-classification) | Binary classification | Tiny feature-heavy dataset that is useful for verifying model behavior on small datasets with many predictors. |

## Testing Coverage

These datasets collectively cover:

- Binary classification
- Multiclass classification
- Regression
- Imbalanced classification
- Small datasets
- Medium-sized datasets
- Mixed numeric and categorical features
- Missing values
- Feature engineering requirements
- Probability outputs and threshold-based evaluation
- Model explainability through feature importance

## Suggested Evaluation Metrics

| Task Type | Suggested Metrics |
|---|---|
| Binary classification | Accuracy, Precision, Recall, F1, ROC-AUC |
| Imbalanced binary classification | Precision, Recall, F1, ROC-AUC, PR-AUC |
| Multiclass classification | Accuracy, Macro F1, Weighted F1, Confusion Matrix |
| Regression | MAE, RMSE, R² |

## UPDATED Project Structure (05.22.25 @ 5:17)
```text
my_project/
├── data/
│   └── titanic/
│       └── train.csv
├── notebooks/
│   └── 01_titanic.ipynb
├── src/
│   ├── __init__.py
│   ├── experiment.py
│   ├── evaluation.py
│   └── models.py
└── results/
```

## OLD Project structure

```text
classical-ML-baseline-code/
├── README.md
├── notebook.ipynb
├── data.xlsx (in application when not using synthetic data)
└── results/
```
