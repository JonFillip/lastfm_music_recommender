import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from collections import Counter
from scipy.sparse import csr_matrix, hstack
from src.utils.logging_utils import get_logger
import multiprocessing as mp
from functools import partial
import gc

logger = get_logger('data_preprocessing')

def load_data_from_bigquery(project_id, dataset_id, table_id):
    try:
        client = bigquery.Client(project=project_id)
        query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{table_id}`
        """
        df = client.query(query).to_dataframe()
        logger.info(f"Data loaded successfully from BigQuery table {project_id}.{dataset_id}.{table_id}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from BigQuery: {e}")
        raise

def robust_string_parser(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        return [item.strip() for item in x.split(',') if item.strip()]
    if isinstance(x, (list, np.ndarray)):
        return [str(item) for item in x if str(item).strip()]
    return [str(x)] if str(x).strip() else []

def preprocess_data(df):
    try:
        if 'album' in df.columns:
            df = df.drop('album', axis=1)
        df['tags'] = df['tags'].apply(robust_string_parser)
        df['similar_tracks'] = df['similar_tracks'].apply(robust_string_parser)
        df['has_tags'] = (df['tags'].apply(len) > 0).astype(int)
        df['has_similar_tracks'] = (df['similar_tracks'].apply(len) > 0).astype(int)
        df['playcount'] = pd.to_numeric(df['playcount'], errors='coerce')
        df['playcount'].fillna(df['playcount'].median(), inplace=True)

        # Additional data quality checks
        logger.info(f"Missing values: {df.isnull().sum()}")
        logger.info(f"Data types: {df.dtypes}")
        logger.info(f"Unique values in categorical columns: {df.select_dtypes(include=['object']).nunique()}")

        return df
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

def one_hot_encode_sparse(series):
    unique_items = set(item for sublist in series for item in sublist)
    item_to_index = {item: i for i, item in enumerate(unique_items)}
    rows, cols = [], []
    for i, sublist in enumerate(series):
        for item in sublist:
            rows.append(i)
            cols.append(item_to_index[item])
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(series), len(unique_items)))

def process_chunk(chunk, tags_vocab, tracks_vocab):
    tags_encoded = one_hot_encode_sparse(chunk['tags'])
    tracks_encoded = one_hot_encode_sparse(chunk['similar_tracks'])
    return hstack([tags_encoded, tracks_encoded])

def impute_data(df, n_neighbors=10, chunk_size=10000):
    numeric_data = df[['playcount']].values

    # Create vocabularies for tags and tracks
    tags_vocab = set(item for sublist in df['tags'] for item in sublist)
    tracks_vocab = set(item for sublist in df['similar_tracks'] for item in sublist)

    # Process data in chunks
    with mp.Pool(mp.cpu_count()) as pool:
        encoded_chunks = pool.map(
            partial(process_chunk, tags_vocab=tags_vocab, tracks_vocab=tracks_vocab),
            [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        )

    # Combine chunks
    encoded_data = hstack(encoded_chunks)

    # Combine numeric and encoded data
    features = hstack([numeric_data, encoded_data])

    # Scale features
    scaler = StandardScaler(with_mean=False)  # Use with_mean=False for sparse data
    features_scaled = scaler.fit_transform(features)

    # Impute using KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_features = imputer.fit_transform(features_scaled)

    # Rescale imputed features
    imputed_features = scaler.inverse_transform(imputed_features)

    # Update original dataframe
    df['playcount'] = imputed_features[:, 0]

    # Convert one-hot encoded back to lists
    tags_start = 1
    tracks_start = tags_start + len(tags_vocab)

    df['tags'] = [
        [tag for tag, val in zip(tags_vocab, row[tags_start:tracks_start]) if val > 0.5]
        for row in imputed_features
    ]
    df['similar_tracks'] = [
        [track for track, val in zip(tracks_vocab, row[tracks_start:]) if val > 0.5]
        for row in imputed_features
    ]

    # Fill empty lists with most common tags/tracks
    global_common_tags = Counter([tag for tags in df['tags'] for tag in tags]).most_common(5)
    global_common_tracks = Counter([track for tracks in df['similar_tracks'] for track in tracks]).most_common(5)

    df.loc[df['tags'].apply(len) == 0, 'tags'] = [tag for tag, _ in global_common_tags]
    df.loc[df['similar_tracks'].apply(len) == 0, 'similar_tracks'] = [track for track, _ in global_common_tracks]

    # Convert lists back to strings
    df['tags'] = df['tags'].apply(lambda x: ', '.join(x) if x else 'Unknown')
    df['similar_tracks'] = df['similar_tracks'].apply(lambda x: ', '.join(x) if x else 'Unknown')

    # Update has_tags and has_similar_tracks
    df['has_tags'] = (df['tags'] != 'Unknown').astype(int)
    df['has_similar_tracks'] = (df['similar_tracks'] != 'Unknown').astype(int)

    return df

def write_to_bigquery(df, project_id, dataset_id, table_id):
    try:
        client = bigquery.Client(project=project_id)
        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            write_disposition="WRITE_TRUNCATE",
        )
        job = client.load_table_from_dataframe(
            df, f"{project_id}.{dataset_id}.{table_id}", job_config=job_config
        )
        job.result()  # Wait for the job to complete
        logger.info(f"Processed data written to BigQuery table {project_id}.{dataset_id}.{table_id}")
    except Exception as e:
        logger.error(f"Error writing data to BigQuery: {e}")
        raise

def main(project_id, input_dataset_id, input_table_id, output_dataset_id, output_table_id):
    try:
        df = load_data_from_bigquery(project_id, input_dataset_id, input_table_id)
        df_processed = preprocess_data(df)
        imputed_data = impute_data(df_processed)
        imputed_data.drop_duplicates(inplace=True)
        write_to_bigquery(imputed_data, project_id, output_dataset_id, output_table_id)
        logger.info("Data processing completed successfully")
    except Exception as e:
        logger.error(f"Error in preprocessing main function: {e}")
        raise

if __name__ == "__main__":
    # These would be replaced by Kubeflow pipeline component inputs
    project_id = "your-project-id"
    input_dataset_id = "lastfm_dataset"
    input_table_id = "raw_top_tracks"
    output_dataset_id = "lastfm_dataset"
    output_table_id = "processed_top_tracks"
    
    main(project_id, input_dataset_id, input_table_id, output_dataset_id, output_table_id)
    logger.info("Preprocessing script executed. This would be replaced by Kubeflow component execution.")
