# src/models.py

'''

This module contains functions to build machine learning models.
- max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
- random_state: Controls the randomness of the estimator. The features are always randomly permuted at each split, even if max_features=n_features. When max_features < n_features, the algorithm will select max_features at random at each split before finding the best split among them. But the best found

'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def build_decision_tree(max_depth=10, random_state=42):
    return DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )



def build_random_forest(n_estimators=200, max_depth=8, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )