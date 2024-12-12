# Data collection
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker="ZOMATO.NS", start_date="2021-01-01", end_date="2024-12-31"):
    # Fetch data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Date'] = stock_data.index
    return stock_data

#Feature Engineering
def add_technical_indicators(data):
    # Adding 50-day Moving Average
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Adding 200-day Moving Average
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Adding Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data.dropna()

# Preparing Data for the Model

from sklearn.preprocessing import MinMaxScaler

def prepare_data(data):
    # Drop unwanted columns and use only 'Close' price
    data = data[['Date', 'Close', 'MA50', 'MA200', 'RSI']]
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'MA50', 'MA200', 'RSI']])
    
    # Prepare the training data (look-back time steps)
    def create_dataset(data, look_back=60):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i-look_back:i, :-1])
            y.append(data[i, -1])
        return np.array(X), np.array(y)
    
    X, y = create_dataset(scaled_data)
    return X, y, scaler

# LSTM Model for Prediction
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predicting the 'Close' price
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Training and Saving the Model
import joblib
from sklearn.metrics import mean_squared_error

def train_model(X_train, y_train, X_test, y_test, scaler):
    model = build_lstm_model(X_train)
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Inverse transform the predictions and actual values to compare them
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error: {rmse}')
    
    # Save the model
    joblib.dump(model, 'lstm_model.pkl')
    return rmse
def train_model(X_train, y_train, X_test, y_test, scaler):
    model = build_lstm_model(X_train)
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Save predictions for the week
    weekly_predictions = scaler.inverse_transform(y_pred[-7:])  # Last 7 predictions
    with open("weekly_predictions.txt", "a") as pred_file:
        pred_file.write(f"Weekly predictions on {datetime.now()}:\n{weekly_predictions}\n")

    return model

# Automating the Data Collection and Training
import time
import git

def commit_changes():
    repo = git.Repo('abizer007/zomato-stock-prediction')
    repo.git.add(A=True)  # Stage all changes
    repo.index.commit(f"Automated commit on {datetime.now()}")
    origin = repo.remote(name='origin')
    origin.push()


def run_daily():
    while True:
        # Fetch the latest data
        data = fetch_data()
        data = add_technical_indicators(data)
        
        # Prepare data and split into training and test sets
        X, y, scaler = prepare_data(data)
        
        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        # Train the model and log performance
        rmse = train_model(X_train, y_train, X_test, y_test, scaler)
        
        # Commit changes to GitHub
        commit_changes()
        
        # Wait for 24 hours before fetching data again
        time.sleep(86400)  # 24 hours in seconds
