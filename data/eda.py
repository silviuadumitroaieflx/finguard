import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# cate tranzactii sunt fraude 
print(df["isFraud"].value_counts())
print("Fraud %:", round(df["isFraud"].mean() * 100, 3), "%")

# fraude vs normale (media si maximul sumelor)
print(df.groupby("isFraud")["amount"].agg(["mean", "max"]))
sample = df.sample(50_000, random_state=42)

# zoom pe un interval relevant
q99 = df["amount"].quantile(0.99)

# grafic general
sample[sample["amount"] < q99]["amount"].hist(bins=80)
plt.xlabel("Amount")
plt.ylabel("Count")
plt.title("Transaction amounts (sample, < 99th percentile)")
plt.show()

# normale vs fraude
normal = df[df["isFraud"] == 0]["amount"]
fraud  = df[df["isFraud"] == 1]["amount"]

plt.hist(normal[normal < q99], bins=60, range=(0, q99), alpha=0.5, label="Normal", density=True)
plt.hist(fraud[fraud < q99],  bins=60, range=(0, q99), alpha=0.6, label="Fraud", color="red", density=True)
plt.xlabel("Amount")
plt.ylabel("Density")
plt.title("Normal vs Fraud amounts (zoomed < 99th pct)")
plt.legend()
plt.show()

# grafic pe scara logaritmica (pentru vizibilitate sporita )
plt.hist(normal, bins=100, alpha=0.4, label="Normal", log=True)
plt.hist(fraud,  bins=100, alpha=0.7, label="Fraud", color="red", log=True)
plt.xlabel("Amount")
plt.ylabel("Count (log scale)")
plt.title("Normal vs Fraud amounts (log scale)")
plt.legend()
plt.show()

# %fraude pe tip de tranzactie
fraud_rate_by_type = (df.groupby("type")["isFraud"].mean() * 100).sort_values(ascending=False)
print("\nFraud rate by transaction type (%):")
print(fraud_rate_by_type)

fraud_rate_by_type.plot(kind="bar")
plt.ylabel("Fraud rate (%)")
plt.title("Fraud rate by transaction type")
plt.show()
