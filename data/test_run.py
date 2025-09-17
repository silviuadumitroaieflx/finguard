import pandas as pd
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
input("Apasă Enter pentru a închide...")
