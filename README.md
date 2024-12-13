# zomato-stock-predictor
# Zomato Stock Price Prediction

This project aims to predict the future stock prices of Zomato (ZOMATO.NS) using historical stock data. The prediction model utilizes an advanced Long Short-Term Memory (LSTM) architecture, which is a type of recurrent neural network (RNN) designed to handle time series data. By leveraging features like moving averages and the Relative Strength Index (RSI), the model forecasts future stock prices, which can be useful for traders, analysts, and other stakeholders in the stock market.

The project is designed to automatically collect and preprocess daily stock data, train a machine learning model, and save predictions on a regular basis. The results and model performance are logged and stored in various files for future reference and analysis.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Folder Structure](#folder-structure)
7. [Model Explanation](#model-explanation)
8. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
9. [Prediction and Logging](#prediction-and-logging)
10. [Scheduled Updates](#scheduled-updates)
11. [Contributing](#contributing)
12. [License](#license)

---

## Project Overview

The Zomato Stock Price Prediction project uses deep learning (LSTM) to forecast the stock prices of Zomato based on historical data and technical indicators. It involves the following steps:

1. **Data Collection**: Collects daily stock data for Zomato.
2. **Data Preprocessing**: Adds technical indicators such as moving averages and RSI to enhance the model's prediction accuracy.
3. **Model Training**: Trains an LSTM model on historical stock data to predict future prices.
4. **Prediction Generation**: Makes predictions for the next 7 days, aggregates them monthly and yearly, and saves them for further analysis.
5. **Logging**: Logs predictions and model performance in different text files and CSV files.
6. **Automatic GitHub Updates**: The project automatically commits and pushes changes to the GitHub repository daily.

---

## Features

- **Stock Data Collection**: 
  - Collects Zomato stock data daily from external sources such as Yahoo Finance or other APIs.
  
- **Technical Indicators**: 
  - Adds technical indicators such as **Simple Moving Averages (SMA)**, **Exponential Moving Averages (EMA)**, and **Relative Strength Index (RSI)** to help the model learn trends in stock prices.

- **Predictive Modeling with LSTM**:
  - Uses **Long Short-Term Memory (LSTM)** neural networks for time-series forecasting.
  - Trains the LSTM model on historical stock data to predict future stock prices.

- **Performance Logging**: 
  - Logs model performance (e.g., loss, accuracy) during training in a `performance.txt` file.
  
- **Prediction Logging**: 
  - Saves weekly, monthly, and yearly stock price predictions in separate text files (`weekly_predictions.txt`, `monthly_predictions.txt`, `yearly_predictions.txt`).

- **Stock Data Logging**:
  - Logs stock data updates (e.g., prediction values and actual stock prices) into a CSV file (`stock_data.csv`).

- **Automatic GitHub Commit**: 
  - The project automatically commits updates to the repository every day, ensuring that the latest predictions and logs are stored in version control.

---

## Requirements

Before running the project, make sure you have the following installed:

- Python 3.6+
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `yfinance`
  - `tensorflow` (for LSTM model)
  - `matplotlib` (for visualization)
  - `sklearn` (for scaling data)
  - `csv`
  - `os`
  - `datetime`

## Model Explanation

The core of this project lies in the LSTM model, a type of recurrent neural network (RNN) well-suited for time-series forecasting. The model is trained on past stock prices and technical indicators, such as moving averages and RSI, to learn the temporal patterns and dependencies in the stock price movement.

## Data Preprocessing

The raw stock data is fetched, and technical indicators like SMA, EMA, and RSI are computed.  
The data is normalized using a Min-Max scaler to ensure that all features are on a similar scale.  
The data is split into training and test sets, and the sequences are prepared for the LSTM model.

## LSTM Model Architecture

The model consists of one or more LSTM layers followed by dense layers for output prediction. The LSTM layers capture the temporal dependencies in the data, while the dense layers output the final prediction.

## Model Training

The model is trained on the prepared data, and the training performance (loss, accuracy) is logged. After training, the model predicts the future stock prices for the next 7 days.

## Data Collection and Preprocessing

The data is collected from Yahoo Finance using the `yfinance` library, which provides an easy-to-use interface for downloading stock data. The data is then preprocessed by:

- Calculating Moving Averages (SMA and EMA) to capture trends in stock prices.
- Computing the Relative Strength Index (RSI) to measure overbought and oversold conditions in the stock.
- Normalizing the data using the Min-Max scaler to ensure consistency and improve model performance.

## Prediction and Logging

After the model is trained, the following steps occur:

### Prediction Generation

The model predicts stock prices for the next 7 days, as well as for aggregated monthly and yearly periods.

### Logging

The predictions are saved in text files (`weekly_predictions.txt`, `monthly_predictions.txt`, and `yearly_predictions.txt`). The performance data and stock data are saved in `performance.txt` and `stock_data.csv` respectively.

### Commit Updates

Updates (predictions and logs) are automatically committed to the GitHub repository, ensuring that the latest results are version-controlled.

## Contributing

We welcome contributions to the project! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push your changes to your fork (`git push origin feature-branch`).
5. Create a pull request to merge your changes into the main repository.
