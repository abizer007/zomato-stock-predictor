import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Fetching the stock data
def fetch_data(ticker):
    data = yf.download(ticker, start='2015-01-01', end='2022-12-31')
    return data

# Adding technical indicators
def add_technical_indicators(df):
    df['SMA'] = talib.SMA(df['Close'], timeperiod=30)
    df['EMA'] = talib.EMA(df['Close'], timeperiod=30)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    return df

# Data preparation
def prepare_data(df):
    df = df.dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Building the LSTM model
def build_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Training the model
def train_model(ticker):
    df = fetch_data(ticker)
    df = add_technical_indicators(df)
    X, y, scaler = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = build_lstm_model(X_train)
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    
    # Save the model and scaler
    model.save(f'{ticker}_lstm_model.h5')
    joblib.dump(scaler, f'{ticker}_scaler.pkl')

# Main
if __name__ == "__main__":
    train_model('ZOMATO.NS')
