import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

st.set_page_config(page_title="FinGuard - Fraud Detector", layout="wide")

MODEL_PATH = "models/logreg_finguard.joblib"
os.makedirs("models", exist_ok=True)
SAMPLE_PATH = "data/sample_paysim_500.csv"

st.title("FinGuard")

col1,col2= st.columns([1,2])
with col1:uploaded = st.file_uploader("Upload CSV",type=["csv"])
use_sample = st.button("Use sample CSV (data/sample_paysim_500.csv)")

with col2:st.write("Model file:",MODEL_PATH)
st.write("use sample for quicker test")


#load df
df= None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    if not os.path.exists(SAMPLE_PATH):
        st.error(f"Sample not found: {SAMPLE_PATH}")
        st.stop()
    df=pd.read_csv(SAMPLE_PATH)
else:
        st.info("Upload a CSV or click Use sample CSV.")
        st.stop()


# step -> hour/day
df["hour"] = df["step"] % 24
df["day"] = df["step"] // 24

# features
features = df[["amount", "oldbalanceOrg", "newbalanceOrig", "type", "isFraud", "hour", "step"]].copy()
features = pd.get_dummies(features, columns=["type"], drop_first=True)
features["balance_diff_orig"] = features["oldbalanceOrg"] - features["amount"] - features["newbalanceOrig"]
features["amount_ratio"] = features["amount"] / (features["oldbalanceOrg"] + 1)
features["flag_insufficient_funds"] = (features["oldbalanceOrg"] < features["amount"]).astype(int)
features["night_tx"] = (features["hour"] <= 4).astype(int)
features["client_tx"] = df.groupby("nameOrig")["step"].transform("count")

for col in ["type_TRANSFER", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT"]:
    if col not in features.columns:
        features[col] = 0

st.subheader("Features preview (first 10 rows)")
st.dataframe(features.head(10))

# var
X = features.drop(columns=["isFraud"], errors="ignore")
y = features["isFraud"].astype(int) if "isFraud" in features.columns else None

#logreg
model = None
if os.path.exists(MODEL_PATH) and not retrain_btn:
    try:
        model = joblib.load(MODEL_PATH)
        st.info("Loaded saved model.")
    except Exception as e:
        st.warning("Could not load saved model, will train a fresh one.")
        model = None

if (model is None) or retrain_btn:
    if y is None or y.nunique() < 2:
        st.warning("Not enough labels to train (need both classes). The app will still show scores if a model exists.")
    else:
        st.info("Training LogisticRegression (quick baseline)...")
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        st.success("Model trained and saved.")

if model is None:
    st.error("Model not available and could not be trained.")
    st.stop()

# top alerts
probas = model.predict_proba(X)[:, 1]
df_out = df.copy()
df_out["fraud_probability"] = probas

st.subheader("Top alerts (first 20)")
top = df_out.sort_values("fraud_probability", ascending=False).head(50)

show_cols = ["step", "hour", "amount", "nameOrig", "nameDest", "fraud_probability"]
st.dataframe(top[show_cols].head(20))

# histogram predicted probs
st.subheader("Fraud probability distribution")
fig, ax = plt.subplots()
ax.hist(probas, bins=50)
ax.set_xlabel("fraud probability")
ax.set_ylabel("count")
st.pyplot(fig)

# evaluation if labels exist 
if "isFraud" in df.columns:
    st.subheader("Evaluation on loaded data")
    y_true = df["isFraud"].astype(int)
    y_pred = (probas >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    st.write("Confusion matrix (array):")
    st.write(cm)
    st.text("Classification report:")
    st.text(classification_report(y_true, y_pred, digits=4, zero_division=0))
    st.write("ROC-AUC:", roc_auc_score(y_true, probas))


        
