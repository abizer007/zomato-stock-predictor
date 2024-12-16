import os
from model import fetch_data, add_technical_indicators, prepare_data, build_lstm_model
from datetime import datetime
import numpy as np
import pandas as pd
import csv

# Fetch and preprocess data
data = fetch_data(ticker="ZOMATO.NS")
data = add_technical_indicators(data)
X, y, scaler = prepare_data(data)

# Build and train the model
model = build_lstm_model(X)
model.fit(X, y, epochs=1, batch_size=32)

# Generate predictions
y_pred = model.predict(X[-7:])

# If the prediction is a single feature, replicate it to match the number of features the scaler expects
if y_pred.shape[1] == 1:  # If the model predicts a single feature
    y_pred = np.repeat(y_pred, 4, axis=1)  # Replicate the single feature 4 times

# Inverse transformation to get the actual predicted values
weekly_predictions = scaler.inverse_transform(y_pred)  # Now this should work correctly

# Save weekly predictions
try:
    with open("weekly_predictions.txt", "a") as pred_file:
        pred_file.write(f"Weekly predictions on {datetime.now()}:\n{weekly_predictions}\n")
    print("Weekly predictions saved successfully.")
except Exception as e:
    print(f"Error writing to weekly_predictions.txt: {e}")

# Cumulative Data (Monthly and Yearly predictions)
def aggregate_predictions(predictions, time_period):
    """
    Aggregates the predictions into a specified time period (e.g., monthly or yearly).
    """
    # Convert predictions into a DataFrame, with each row as a daily prediction
    df_pred = pd.DataFrame(predictions, columns=[f"Prediction_{i+1}" for i in range(predictions.shape[1])])
    
    # Add the date for each prediction (using a range of dates for the last 7 days)
    dates = pd.date_range(end=datetime.today(), periods=len(predictions), freq='D')
    df_pred['Date'] = dates
    df_pred.set_index('Date', inplace=True)
    
    # Aggregate by resampling based on the requested time period
    if time_period == 'monthly':
        aggregated_predictions = df_pred.resample('M').sum()  # Summing the daily predictions to get monthly predictions
    elif time_period == 'yearly':
        aggregated_predictions = df_pred.resample('Y').sum()  # Summing the daily predictions to get yearly predictions
    else:
        raise ValueError("Invalid time period. Please use 'monthly' or 'yearly'.")
    
    return aggregated_predictions

# Aggregate predictions for monthly and yearly predictions
monthly_predictions = aggregate_predictions(weekly_predictions, 'monthly')
yearly_predictions = aggregate_predictions(weekly_predictions, 'yearly')

# Save monthly predictions
try:
    with open("monthly_predictions.txt", "a") as pred_file:
        pred_file.write(f"Monthly predictions on {datetime.now()}:\n{monthly_predictions}\n")
    print("Monthly predictions saved successfully.")
except Exception as e:
    print(f"Error writing to monthly_predictions.txt: {e}")

# Save yearly predictions
try:
    with open("yearly_predictions.txt", "a") as pred_file:
        pred_file.write(f"Yearly predictions on {datetime.now()}:\n{yearly_predictions}\n")
    print("Yearly predictions saved successfully.")
except Exception as e:
    print(f"Error writing to yearly_predictions.txt: {e}")

# Example of writing to performance.txt
try:
    with open("performance.txt", "w") as f:
        f.write("Performance data for today's stock prediction\n")
    print("performance.txt created successfully.")
except Exception as e:
    print(f"Error writing to performance.txt: {e}")

# Same for other files like stock_data.csv
try:
    with open("stock_data.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d"), "1000", "Prediction Value"])  # Adjusted as an example
    print("stock_data.csv updated successfully.")
except Exception as e:
    print(f"Error writing to stock_data.csv: {e}")

# Check if the files exist after writing
files_to_check = ["performance.txt", "weekly_predictions.txt", "monthly_predictions.txt", "yearly_predictions.txt", "stock_data.csv"]
for file in files_to_check:
    if os.path.exists(file):
        print(f"{file} created successfully.")
    else:
        print(f"Failed to create {file}."
