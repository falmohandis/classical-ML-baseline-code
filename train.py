# classical-ML-baseline-code/train.py

import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


# ----------------------------
# 1. Create synthetic patient-like data
# ----------------------------
X, y = make_classification(
    n_samples=1000,
    n_features=12,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

x

df = pd.DataFrame(X, columns=feature_names)
df["disease_risk"] = y

print(df.head())


# ----------------------------
# 2. Split data
# ----------------------------
X = df.drop(columns=["disease_risk"])
y = df["disease_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ----------------------------
# 3. Train Decision Tree
# ----------------------------
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

dt_preds = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_preds)

print("\nDecision Tree Results")
print("Accuracy:", dt_accuracy)
print(confusion_matrix(y_test, dt_preds))
print(classification_report(y_test, dt_preds))


# ----------------------------
# 4. Train Random Forest
# ----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_preds)

print("\nRandom Forest Results")
print("Accuracy:", rf_accuracy)
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))


# ----------------------------
# 5. Feature importance
# ----------------------------
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance")
print(importance_df)


# ----------------------------
# 6. Save outputs
# ----------------------------
importance_df.to_csv("results/feature_importance.csv", index=False)

with open("results/model_summary.txt", "w") as f:
    f.write("Classical ML Baseline for Patient Risk Prediction\n")
    f.write("Models compared: Decision Tree and Random Forest\n")
    f.write(f"Decision Tree Accuracy: {dt_accuracy:.4f}\n")
    f.write(f"Random Forest Accuracy: {rf_accuracy:.4f}\n")

print("\nSaved results to the results/ folder.")
