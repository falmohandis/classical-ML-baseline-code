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


## Project structure

```text
classical-ML-baseline-code/
├── README.md
├── notebook.ipynb
├── data.xlsx (in application when not using synthetic data)
└── results/
```
