import pandas as pd
import joblib 

df = pd.read_csv("data/sp500_features.csv")

for col in ["return", "ma5", "ma10", "volatility"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(inplace=True)

X_latest = df[["return", "ma5", "ma10", "volatility"]].tail(1)

model = joblib.load("data/rf_model.pkl")

pred = model.predict(X_latest)
print(f"Predicted 5-day return: {pred[0]:.4f}")
