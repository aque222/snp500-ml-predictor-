# snp500-ml-predictor-
Overview:
Built a machine learning model to forecast 5-day returns of the S&P 500 index using historical price-based indicators. This project demonstrates the use of feature engineering and regression modelling in a financial context.

Key Steps and Results: 
- collected historical S&P 500 data with yfinance.
- engineered predictive features such as moving averages, volatility, and momentum.
- Trained a Random Forest Regressor to predict short-term returns.
- Achieved a R^2 of ~0.12 on the test set (capturing some market structure despite noise). 
- Visualised predicted vs actual returns, showing the model's ability to track directional trends, though with limited accuracy (reflecting the inherent difficulty of short-term market prediction).

Impact:
This project demonstrates the challenges of equity return prediction while showcasing the ability to design end-to-end ML pipelines, evaluate models with appropriate metrics, and communicate results transparently. 

Skills demonstrated: 
- Python
- yfinance
- scikit-learn
- regression modeling
- financial feature engineering
- performance evaluation
- data visualisation 
