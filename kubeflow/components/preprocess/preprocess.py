import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from src.utils.logging_utils import get_logger
from src.data_processing.data_preprocess import load_data, preprocess_data, split_data

logger = get_logger('data_preprocessing')

def main(input_file_path, output_train_path, output_val_path, output_test_path):
    try:
        # Load data
        df = load_data(input_file_path)
        
        # Preprocess data
        df_processed, mlb = preprocess_data(df)
        
        # Split data
        train, val, test = split_data(df_processed)
        
        # Save processed datasets
        train.to_csv(output_train_path, index=False)
        val.to_csv(output_val_path, index=False)
        test.to_csv(output_test_path, index=False)
        
        logger.info(f"Preprocessed data saved to {output_train_path}, {output_val_path}, and {output_test_path}")
        
        return mlb  # Return the MultiLabelBinarizer for future use if needed
    except Exception as e:
        logger.error(f"Error in preprocessing main function: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for music recommender')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--output_train', type=str, required=True, help='Path to save the preprocessed training data')
    parser.add_argument('--output_val', type=str, required=True, help='Path to save the preprocessed validation data')
    parser.add_argument('--output_test', type=str, required=True, help='Path to save the preprocessed test data')
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_train, args.output_val, args.output_test)