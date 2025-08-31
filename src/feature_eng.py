import pandas as pd 

df = pd.read_csv("data/sp500.csv")

df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

df["return"] = df["Close"].pct_change()
df["future_5d_return"] = df["Close"].pct_change(5).shift(-5)

df["ma5"] = df["Close"].rolling(5).mean()
df["ma10"] = df["Close"].rolling(10).mean()
df["volatility"] = df["return"].rolling(10).std()

df.dropna(inplace=True)

df.to_csv("data/sp500_features.csv", index=False)
print("Features saved to data/sp500_features.csv")