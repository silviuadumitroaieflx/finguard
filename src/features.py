import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df_full = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")
df = df_full.sample(10000, random_state=42).reset_index(drop=True)
df.to_csv("data/sample_paysim_500.csv", index=False)

#step
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

   
X = features.drop(columns=["isFraud"])
y = features["isFraud"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
