import pandas as pd
import os
from model import run_model

# Function to log model data to file
def log_data_to_file(stock_data, predictions, filename='stock_data.txt'):
    with open(filename, 'a') as f:
        f.write(f"Timestamp: {pd.to_datetime('now')}\n")
        f.write("Stock Data:\n")
        f.write(stock_data.to_string())
        f.write("\nPredictions:\n")
        f.write(str(predictions))
        f.write("\n\n")

# Run model and log predictions
def run_and_log():
    model, rmse, mae = run_model()
    
    # Log stock data and predictions
    stock_data = pd.read_csv('stock_data.csv')  # Assuming you're saving stock data to CSV
    predictions = model.predict(stock_data[['SMA', 'EMA', 'RSI', 'MACD', 'ATR']])
    
    log_data_to_file(stock_data, predictions)

if __name__ == '__main__':
    run_and_log()
