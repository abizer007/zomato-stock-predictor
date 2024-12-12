import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from model import fetch_data, add_technical_indicators, prepare_data, build_lstm_model

# Test if fetch_data returns a DataFrame with expected columns
def test_fetch_data():
    data = fetch_data(ticker="ZOMATO.NS")
    assert isinstance(data, pd.DataFrame)
    assert 'Date' in data.columns
    assert 'Close' in data.columns

# Test if technical indicators are added to the dataset correctly
def test_add_technical_indicators():
    data = fetch_data(ticker="ZOMATO.NS")
    data = add_technical_indicators(data)
    assert 'MA50' in data.columns
    assert 'MA200' in data.columns
    assert 'RSI' in data.columns

# Test if data preparation function works as expected
def test_prepare_data():
    data = fetch_data(ticker="ZOMATO.NS")
    data = add_technical_indicators(data)
    X, y, scaler = prepare_data(data)
    assert X.shape[0] == len(data) - 60
    assert X.shape[1] == 60
    assert len(y) == len(X)

# Test if the LSTM model can be built and compiled
def test_build_lstm_model():
    data = fetch_data(ticker="ZOMATO.NS")
    data = add_technical_indicators(data)
    X, y, _ = prepare_data(data)
    model = build_lstm_model(X)
    assert model is not None
    assert len(model.layers) > 0

# Test if weekly predictions file is being written correctly
def test_weekly_predictions():
    data = fetch_data(ticker="ZOMATO.NS")
    data = add_technical_indicators(data)
    X, y, scaler = prepare_data(data)
    model = build_lstm_model(X)
    model.fit(X, y, epochs=1, batch_size=32)
    y_pred = model.predict(X[-7:])
    weekly_predictions = scaler.inverse_transform(y_pred)
    
    with open("weekly_predictions.txt", "a") as pred_file:
        pred_file.write(f"Weekly predictions on {datetime.now()}:\n{weekly_predictions}\n")
    
    with open("weekly_predictions.txt", "r") as pred_file:
        content = pred_file.read()
        assert "Weekly predictions on" in content
        assert len(content.split("\n")) > 1

