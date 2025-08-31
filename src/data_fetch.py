import os 
import yfinance as yf 
import pandas as pd

os.makedirs("data", exist_ok=True)

df= yf.download("^GSPC", start="2010-01-01", end="2025-01-01")
df.reset_index(inplace=True)

df.to_csv("data/sp500.csv", index=False)
print("Data saved to data/sp500.csv")
