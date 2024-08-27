import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import os
from IPython.display import display

def plot_correlation_map(df):
    new_df = df.copy()

    # Check correlations
    correlation_matrix = new_df.select_dtypes(include=[np.number]).corr()

    # Visualize correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.show()


def save_data_artifact(df, base_path, filename):
    if df is not None:
            display(df.head())  # Print the first few rows of the DataFrame to verify

            # Ensure the directory exists
            output_dir = os.path.join(os.getcwd(), base_path)
            os.makedirs(output_dir, exist_ok=True)

            # Save DataFrame to the specified path
            output_path = os.path.join(output_dir, filename)
            df.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
    else:
        print("Failed to fetch data, DataFrame is None")
