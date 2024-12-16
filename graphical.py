import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Read data from file
def read_predictions(file_path):
    data = pd.read_csv(file_path, header=None, delimiter=" ", engine='python')
    return data.values

# Function to plot the data
def plot_predictions(data):
    time_stamps = [f"Prediction {i+1}" for i in range(data.shape[0])]
    
    plt.figure(figsize=(10, 6))
    
    # Plot each column (you can change this according to which data you want to visualize)
    for i in range(data.shape[1]):
        plt.plot(time_stamps, data[:, i], label=f"Prediction {i+1}")
    
    plt.title("Weekly Predictions")
    plt.xlabel("Time Stamps")
    plt.ylabel("Prediction Values")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    
    # Create 'logs' directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Save the plot as a PNG file in the 'logs' folder
    plot_path = 'logs/predictions_plot.png'
    plt.savefig(plot_path)
    plt.close()

# Main function to read the data and plot
def main():
    file_path = 'weekly_predictions.txt'
    data = read_predictions(file_path)
    plot_predictions(data)

if __name__ == "__main__":
    main()
