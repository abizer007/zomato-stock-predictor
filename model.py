import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import talib

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Feature engineering: Add technical indicators
def add_technical_indicators(df):
    df['SMA'] = df['Close'].rolling(window=14).mean()  # Simple Moving Average
    df['EMA'] = df['Close'].ewm(span=14, adjust=False).mean()  # Exponential Moving Average
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)  # Relative Strength Index
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)  # MACD
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)  # Average True Range
    df = df.dropna()  # Drop rows with NaN values
    return df

# Prepare features and target variable
def prepare_data(df):
    X = df[['SMA', 'EMA', 'RSI', 'MACD', 'ATR']]  # Features
    y = df['Close']  # Target: Closing price
    return X, y

# Train model with hyperparameter optimization (GridSearchCV)
def train_model(X, y):
    model = RandomForestRegressor(random_state=42)
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X, y, cv=3)
    print(f"Cross-validation scores: {cv_scores}")
    
    return best_model

# Model evaluation
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    return rmse, mae

# Main function to run the model
def run_model(ticker='ZOMATO.NS', start_date='2023-01-01', end_date='2027-12-31'):
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data = add_technical_indicators(stock_data)
    
    X, y = prepare_data(stock_data)
    model = train_model(X, y)
    rmse, mae = evaluate_model(model, X, y)
    
    return model, rmse, mae

if __name__ == '__main__':
    model, rmse, mae = run_model()

