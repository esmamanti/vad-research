"""Scoring: how well does one feature, or a set of features, answer a yes/no question?

'target' is the name of the label column: "audio" (is there any sound?) or "speech" (is there speech?).
Everything is fitted on the TRAIN table and scored on the TEST table (different recordings).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def scores(true, predicted):
    """Compare predictions (0/1) with the true labels (0/1)."""
    true, predicted = np.asarray(true), np.asarray(predicted)
    hit = np.mean(predicted[true == 1] == 1)          # share of the "yes" frames that we found
    reject = np.mean(predicted[true == 0] == 0)       # share of the "no" frames that we rejected
    return {
        "accuracy": np.mean(predicted == true),       # share of all frames that are correct
        "balanced_acc": (hit + reject) / 2,           # average of the two, fair when one class is rare
        "miss_rate": 1 - hit,                         # "yes" frames we called "no"
        "false_alarm": 1 - reject,                    # "no" frames we called "yes"
    }


def best_threshold(values, labels):
    """Find the cut point of ONE feature that separates the two classes best on the training data.

    Returns (threshold, direction). direction = +1 means "above the threshold = yes",
    direction = -1 means "below the threshold = yes".
    """
    candidates = np.quantile(values, np.linspace(0.005, 0.995, 200))
    best = (-1.0, None, None)
    for threshold in candidates:
        for direction in (+1, -1):
            predicted = (direction * (values - threshold) > 0).astype(int)
            quality = scores(labels, predicted)["balanced_acc"]
            if quality > best[0]:
                best = (quality, threshold, direction)
    return best[1], best[2]


def test_single_feature(train, test, feature, target):
    """Threshold one feature (cut point learned on train) and score it on test."""
    threshold, direction = best_threshold(train[feature].values, train[target].values)
    predicted = (direction * (test[feature].values - threshold) > 0).astype(int)
    result = scores(test[target].values, predicted)
    result["threshold"] = threshold
    result["rule"] = "above = yes" if direction == 1 else "below = yes"
    return result


def fit_model(train, features, target):
    """Logistic regression: a weighted sum of the features, turned into a yes/no decision.

    StandardScaler puts all features on the same scale first (energy is in dB, centroid in Hz).
    class_weight='balanced' stops the model from simply favouring the bigger class.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(train[features].values, train[target].values)
    return model


def test_feature_set(train, test, features, target):
    """Fit a model on a set of features and score it on test."""
    model = fit_model(train, features, target)
    return scores(test[target].values, model.predict(test[features].values))