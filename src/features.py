from __future__ import annotations
import argparse
import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
classification_report,
confusion_matrix,
roc_auc_score,
ConfusionMatrixDisplay,
RocCurveDisplay,
PrecisionRecallDisplay,
)

#data load
df = pd.read_csv("data/sample_paysim_500.csv")


df["hour"] = df["step"] % 24
df["day"]  = df["step"] // 24


#features
features = df[["amount", "oldbalanceOrg", "newbalanceOrig", "type", "isFraud", "hour", "step"]].copy()

features = pd.get_dummies(features, columns=["type"], drop_first=True)

features["balance_diff_orig"] = features["oldbalanceOrg"] - features["amount"] - features["newbalanceOrig"]

features["amount_ratio"] = features["amount"] / (features["oldbalanceOrg"] + 1)

features["flag_insufficient_funds"] = (features["oldbalanceOrg"] < features["amount"]).astype(int)

features["night_tx"] = (features["hour"] <= 4).astype(int)

features["client_tx"] = df.groupby("nameOrig")["step"].transform("count")

frauds_per_hour = df[df["isFraud"] == 1]["hour"].value_counts().sort_index()

print("Frauds per hour (sample):")
print(frauds_per_hour)

print("\nFeatures sample:")
print(features.head())
print("\nShapes -> features:", features.shape)

 #var  
X = features.drop(columns=["isFraud"])
y = features["isFraud"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced" 
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]


ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
plt.title("Precision–Recall Curve")
plt.show()

#rf
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("RF - Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("RF - ROC Curve")
plt.show()

PrecisionRecallDisplay.from_estimator(rf, X_test, y_test)
plt.title("RF - Precision–Recall Curve")
plt.show()



imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop feature importances (RF):\n", imp.head(10))

plt.figure(figsize=(8,5))
imp.head(10).iloc[::-1].plot(kind="barh")
plt.title("Random Forest – Top 10 feature importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


#rf
threshold = 0.15
y_pred_thr = (y_prob_rf >= threshold).astype(int)
print(f"\nWith threshold={threshold}:")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_thr))
print(classification_report(y_test, y_pred_thr, digits=4, zero_division=0))

print("\n==== RandomForest ====")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification report:\n", classification_report(y_test, y_pred_rf, digits=4, zero_division=0))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_rf), 4))

#logreg
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
