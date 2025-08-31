import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv("data/sp500_features.csv")

for col in ["return", "ma5", "ma10", "volatility", "future_5d_return"]: 
    df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

features =["return", "ma5", "ma10", "volatility"]
X = df[features]
y = df["future_5d_return"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.6f}")

joblib.dump(model, "data/rf_model.pkl")
print("Model saved to data/rf_model.pkl")

