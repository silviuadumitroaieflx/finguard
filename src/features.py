import pandas as pd
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

features = df[["amount", "oldbalanceOrg", "newbalanceOrg", "type", "isFraud"]]
print(features.head())


