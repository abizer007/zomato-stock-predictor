from model import fetch_data, add_technical_indicators, prepare_data, build_lstm_model
from datetime import datetime
import numpy as np

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

# Save predictions to a file
with open("weekly_predictions.txt", "a") as pred_file:
    pred_file.write(f"Weekly predictions on {datetime.now()}:\n{weekly_predictions}\n")
