import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from src.utils.logging_utils import get_logger

logger = get_logger('data_preprocessing')

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def preprocess_data(df):
    try:
        # Convert playcount to numeric and handle any non-numeric values
        df['playcount'] = pd.to_numeric(df['playcount'], errors='coerce')
        df['playcount'].fillna(df['playcount'].median(), inplace=True)

        # Normalize playcount
        scaler = StandardScaler()
        df['playcount_normalized'] = scaler.fit_transform(df[['playcount']])

        # Process tags
        mlb = MultiLabelBinarizer()
        tags_encoded = mlb.fit_transform(df['tags'].str.split(', '))
        tag_columns = [f'tag_{tag}' for tag in mlb.classes_]
        df_tags = pd.DataFrame(tags_encoded, columns=tag_columns)

        # Combine features
        df_processed = pd.concat([df[['name', 'artist', 'album']], df['playcount_normalized'], df_tags], axis=1)

        logger.info("Data preprocessing completed successfully")
        return df_processed, mlb
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

def split_data(df, test_size=0.2, val_size=0.2):
    try:
        # First, split into train+val and test
        train_val, test = train_test_split(df, test_size=test_size, random_state=42)
        
        # Then split train+val into train and val
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
        
        logger.info(f"Data split completed. Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
        return train, val, test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise

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
    # This section would be replaced by Kubeflow pipeline component inputs
    # For now, we'll just log a message
    logger.info("Preprocessing script executed. This would be replaced by Kubeflow component execution.")
