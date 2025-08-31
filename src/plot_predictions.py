import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("data/sp500_features.csv")

for col in ["return", "ma5", "volatility", "future_5d_return"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(inplace=True)

X= df[["return", "ma5", "ma10", "volatility"]]
y_actual = df["future_5d_return"]

model = joblib.load("data/rf_model.pkl")

y_pred = model.predict(X)

N = 200
plt.figure(figsize=(12,6))
plt.plot(y_actual.values[-N:], label="Actual 5-day Return")
plt.plot(y_pred[-N:], label="Predicted 5-day Return")
plt.title(f"Predcited vs Actual 5-day S&P 500 Returns (last{N} days)")
plt.xlabel("Time (days)")
plt.ylabel("Return")
plt.legend()
plt.savefig("data/predicted_vs_actual.png")
plt.show()


