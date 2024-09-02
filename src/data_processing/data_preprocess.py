import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from src.utils.logging_utils import get_logger
from sklearn.impute import KNNImputer
from collections import Counter
import gc
from scipy.sparse import csr_matrix, hstack

logger = get_logger('data_preprocessing')

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def robust_string_parser(x):
    if pd.isna(x):
        return []  # Return empty list instead of np.nan
    if isinstance(x, str):
        return [item.strip() for item in x.split(',') if item.strip()]
    if isinstance(x, (list, np.ndarray)):
        return [str(item) for item in x if str(item).strip()]
    return [str(x)] if str(x).strip() else []

def preprocess_data(df):
    try:

        # Drop the 'album' column as it's all NaN
        df = df.drop('album', axis=1)
    
        # Parse 'tags' and 'similar_tracks' columns
        df['tags'] = df['tags'].apply(robust_string_parser)
        df['similar_tracks'] = df['similar_tracks'].apply(robust_string_parser)
        
        # Create binary indicators for missing values
        df['has_tags'] = (df['tags'].apply(len) > 0).astype(int)
        df['has_similar_tracks'] = (df['similar_tracks'].apply(len) > 0).astype(int)
        
        # Convert playcount to numeric and handle any non-numeric values
        df['playcount'] = pd.to_numeric(df['playcount'], errors='coerce')
        df['playcount'].fillna(df['playcount'].median(), inplace=True)

        return df
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise


def one_hot_encode(series):
    unique_items = set(item for sublist in series for item in sublist)
    return pd.DataFrame([[1 if item in sublist else 0 for item in unique_items] for sublist in series],
                        columns=list(unique_items))


def impute_data(df, n_neighbors=10):
    # Prepare numeric data
    numeric_data = df[['playcount']].copy()
    
    # One-hot encode tags and similar_tracks
    tags_encoded = one_hot_encode(df['tags'])
    tracks_encoded = one_hot_encode(df['similar_tracks'])
    
    # Combine all features
    features = pd.concat([numeric_data, tags_encoded, tracks_encoded], axis=1)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Impute using KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_features = imputer.fit_transform(features_scaled)
    
    # Rescale imputed features
    imputed_features = scaler.inverse_transform(imputed_features)
    
    # Reconstruct dataframe
    imputed_df = pd.DataFrame(imputed_features, columns=features.columns, index=df.index)
    
    # Update original dataframe
    df['playcount'] = imputed_df['playcount']
    
    # Convert one-hot encoded back to lists
    tags_columns = tags_encoded.columns
    tracks_columns = tracks_encoded.columns
    
    df['tags'] = imputed_df[tags_columns].apply(lambda row: [col for col, val in row.items() if val > 0.5], axis=1)
    df['similar_tracks'] = imputed_df[tracks_columns].apply(lambda row: [col for col, val in row.items() if val > 0.5], axis=1)
    
    # Function to get most common tags/tracks for an artist
    def get_most_common(artist, column, n=3):
        artist_data = df[df['artist'] == artist][column]
        all_items = [item for sublist in artist_data for item in sublist if sublist]
        return Counter(all_items).most_common(n)
    
    # Get global most common tags and tracks
    global_common_tags = Counter([tag for tags in df['tags'] for tag in tags]).most_common(5)
    global_common_tracks = Counter([track for tracks in df['similar_tracks'] for track in tracks]).most_common(5)
    
    # Fill empty lists with most common tags/tracks for the artist or global common tags/tracks
    for idx, row in df.iterrows():
        if not row['tags']:
            common_tags = get_most_common(row['artist'], 'tags')
            if common_tags:
                df.at[idx, 'tags'] = [tag for tag, _ in common_tags]
            else:
                df.at[idx, 'tags'] = [tag for tag, _ in global_common_tags]
        
        if not row['similar_tracks']:
            common_tracks = get_most_common(row['artist'], 'similar_tracks')
            if common_tracks:
                df.at[idx, 'similar_tracks'] = [track for track, _ in common_tracks]
            else:
                df.at[idx, 'similar_tracks'] = [track for track, _ in global_common_tracks]
    
    # Convert lists back to strings
    df['tags'] = df['tags'].apply(lambda x: ', '.join(x) if x else 'Unknown')
    df['similar_tracks'] = df['similar_tracks'].apply(lambda x: ', '.join(x) if x else 'Unknown')
    
    # Update has_tags and has_similar_tracks
    df['has_tags'] = (df['tags'] != 'Unknown').astype(int)
    df['has_similar_tracks'] = (df['similar_tracks'] != 'Unknown').astype(int)
    
    return df

def prepare_data(preprocessed_df, original_df):
    X = preprocessed_df.values
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(original_df['similar_tracks'].str.split(','))
    track_names = original_df['name'].values
    
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, track_names, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, names_train, names_test, scaler, mlb


def main(input_file_path, output_imputed_path):
    try:
        # Load data
        df = load_data(input_file_path)
        
        # Preprocess data
        df_processed = preprocess_data(df)

        # Imputed data for missing values
        imputed_data = impute_data(df_processed)
        imputed_data.drop_duplicates(inplace=True)

        imputed_data.to_csv(output_imputed_path, index=False)
        logger.info(f"Preprocessed data saved to {output_imputed_path}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing main function: {e}")
        raise

if __name__ == "__main__":
    # This section would be replaced by Kubeflow pipeline component inputs
    # For now, we'll just log a message
    logger.info("Preprocessing script executed. This would be replaced by Kubeflow component execution.")
