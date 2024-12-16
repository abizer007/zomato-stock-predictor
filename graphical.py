import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging

# Configure logging to capture and save error messages
logging.basicConfig(filename='logs/data_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Read data from file and validate it
def read_predictions(file_path):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None

        # Read the file and handle multiple spaces
        data = pd.read_csv(file_path, header=None, delimiter=r"\s+", engine='python')

        # Check for empty data
        if data.empty:
            logging.error(f"The file {file_path} is empty.")
            return None

        # Ensure all data is numeric and replace NaN values with zeros or drop rows
        data = data.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
        if data.isnull().values.any():
            logging.warning(f"Missing values found in the file {file_path}. These will be removed.")
        data = data.dropna()  # Drop rows with NaN values
        
        # Validate that the data has at least one row and column to avoid plotting empty data
        if data.shape[0] == 0 or data.shape[1] == 0:
            logging.error(f"Invalid data in {file_path}: No valid rows/columns after cleaning.")
            return None

        logging.info(f"Data read successfully from {file_path}")
        return data.values
    except Exception as e:
        logging.error(f"Error reading the data file: {e}")
        return None

# Plotting function
def plot_predictions(data):
    if data is None or data.size == 0:
        logging.warning("No valid data to plot.")
        return
    
    time_stamps = [f"Prediction {i+1}" for i in range(data.shape[0])]
    
    plt.figure(figsize=(10, 6))
    
    # Plot each column
    for i in range(data.shape[1]):
        plt.plot(time_stamps, data[:, i], label=f"Prediction {i+1}")
    
    plt.title("Weekly Predictions")
    plt.xlabel("Time Stamps")
    plt.ylabel("Prediction Values")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)

    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Save the plot
    plot_path = 'logs/predictions_plot.png'
    plt.savefig(plot_path)
    logging.info(f"Plot saved at {plot_path}")
    plt.close()

# Main function to orchestrate the reading and plotting
def main():
    file_path = 'weekly_predictions.txt'
    data = read_predictions(file_path)
    
    # If data is valid, plot the predictions
    plot_predictions(data)

if __name__ == "__main__":
    main()
