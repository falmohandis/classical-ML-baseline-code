# src/data.py
from importlib.resources import path

import pandas as pd
from sklearn.datasets import make_classification


'''

# Understanding the hyperparameters of the make_classification function:
- n_samples=20000, # This is the number of samples we want to generate for the dummy data.

- n_features=12, # This is the number of features we want to generate for the dummy data.

- n_informative=6, # This is the number of "informative features" we want to generate for the dummy data. 
# Informative features are basically those that are actually useful for predicting the target variable.

- n_redundant=2, # This is the number of redundant features we want to generate for the dummy data. 
# Redundant features are those that are highly correlated with other features, and hence hold no predictive value.

- n_classes=5, # This is the number of "classes" aka unique groups we want in the dummy data.
- random_state=42 # This is the random state for reproducibility, i.e. if you run this code multiple times, you will get the same results each time.


'''


# src/data.py

import pandas as pd
from sklearn.datasets import make_classification


def create_dummy_patient_data(
    n_samples=20000,
    n_features=12,
    n_informative=6,
    n_redundant=2,
    n_classes=5,
    random_state=42
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )

    feature_names = [
        "age", "heart_rate", "systolic_bp", "diastolic_bp",
        "spo2", "resp_rate", "glucose", "lactate",
        "creatinine", "hemoglobin", "wbc", "platelets"
    ]

    df = pd.DataFrame(X, columns=feature_names)
    df["disease_risk"] = y

    classes = sorted(df["disease_risk"].unique())

    return df, feature_names, classes



def load_csv_data(file_path):
    return pd.read_csv(file_path)