import os
import yaml
import time
import joblib
import pandas as pd
import numpy as np
from google.cloud import bigquery, aiplatform
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from src.utils.logging_utils import setup_logger

logger = setup_logger('data_prep')

def load_config() -> Dict[str, Any]:
    with open('configs/pipeline_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data_from_bigquery(project_id: str, dataset_id: str, table_id: str) -> pd.DataFrame:
    """
    Load data from BigQuery table into a pandas DataFrame.
    Uses partitioning and clustering for optimization.
    """
    client = bigquery.Client(project=project_id)
    
    # Assuming the table is partitioned by date and clustered by artist
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE _PARTITIONDATE = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
    ORDER BY artist, name
    """
    
    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        use_legacy_sql=False,
        priority=bigquery.QueryPriority.BATCH
    )
    
    logger.info(f"Loading data from BigQuery table: {project_id}.{dataset_id}.{table_id}")
    df = client.query(query, job_config=job_config).to_dataframe()
    logger.info(f"Loaded {len(df)} rows from BigQuery")
    return df

def prepare_data(preprocessed_df: pd.DataFrame, original_df: pd.DataFrame) -> Tuple:
    """
    Prepare data for model training and testing.
    """
    logger.info("Preparing data for model training and testing")
    
    X = preprocessed_df.drop(['name', 'artist', 'tags', 'similar_tracks', 'playcount'], axis=1, errors='ignore').values
    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(original_df['similar_tracks'].str.split(','))
    track_names = original_df['name'].values
    
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, track_names, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Prepared data shapes: X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, names_train, names_test, scaler, mlb

def save_prepared_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
                        names_train: np.ndarray, names_test: np.ndarray, scaler: StandardScaler, 
                        mlb: MultiLabelBinarizer, output_dir: str):
    """
    Save prepared data and preprocessing objects to files.
    """
    logger.info(f"Saving prepared data to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(output_dir, 'names_train.npy'), names_train)
    np.save(os.path.join(output_dir, 'names_test.npy'), names_test)
    
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(mlb, os.path.join(output_dir, 'multilabel_binarizer.joblib'))

def create_and_populate_feature_store(project_id: str, region: str, feature_store_id: str, entity_type_id: str, df: pd.DataFrame):
    """
    Create and populate a Vertex AI Feature Store.
    """
    aiplatform.init(project=project_id, location=region)

    # Create a feature store
    fs = aiplatform.FeatureStore.create(
        feature_store_id=feature_store_id,
        online_store_fixed_node_count=1,
        sync=True
    )

    # Create an entity type
    entity_type = fs.create_entity_type(
        entity_type_id=entity_type_id,
        description="Music track features"
    )

    # Define features
    features = {
        "artist": "STRING",
        "name": "STRING",
        "tags": "STRING",
        "similar_tracks": "STRING",
        "playcount": "INT64",
        # Add other features as needed
    }

    # Create features
    for feature_id, feature_type in features.items():
        entity_type.create_feature(
            feature_id=feature_id,
            value_type=feature_type,
            description=f"Feature: {feature_id}"
        )

    # Prepare data for ingestion
    feature_time = int(time.time())
    entities = df.to_dict(orient="records")
    for entity in entities:
        entity["feature_time"] = feature_time

    # Ingest feature values
    entity_type.ingest(
        entity_ids=df.index.tolist(),
        feature_time=feature_time,
        features=entities,
        worker_count=10
    )

    logger.info(f"Created and populated feature store: {feature_store_id}")

def main(project_id: str, preprocessed_dataset_id: str, preprocessed_table_id: str, 
            original_dataset_id: str, original_table_id: str, output_dir: str,
            region: str, feature_store_id: str, entity_type_id: str):
    try:
        logger.info("Starting data preparation process")
        
        preprocessed_df = load_data_from_bigquery(project_id, preprocessed_dataset_id, preprocessed_table_id)
        original_df = load_data_from_bigquery(project_id, original_dataset_id, original_table_id)
        
        X_train, X_test, y_train, y_test, names_train, names_test, scaler, mlb = prepare_data(preprocessed_df, original_df)
        
        save_prepared_data(X_train, X_test, y_train, y_test, names_train, names_test, scaler, mlb, output_dir)
        
        # Create and populate feature store
        create_and_populate_feature_store(project_id, region, feature_store_id, entity_type_id, preprocessed_df)
        
        logger.info("Data preparation and feature store population completed successfully")
    except Exception as e:
        logger.error(f"Error in data preparation process: {e}")
        raise

if __name__ == '__main__':
    config = load_config()
    project_id = config['project']['id']
    region = config['project']['region']
    preprocessed_dataset_id = config['bigquery']['preprocessed_dataset_id']
    preprocessed_table_id = config['bigquery']['preprocessed_table_id']
    original_dataset_id = config['bigquery']['original_dataset_id']
    original_table_id = config['bigquery']['original_table_id']
    output_dir = config['data']['prepared_data_dir']
    feature_store_id = config['feature_store']['id']
    entity_type_id = config['feature_store']['entity_type_id']
    
    main(project_id, preprocessed_dataset_id, preprocessed_table_id, 
            original_dataset_id, original_table_id, output_dir,
            region, feature_store_id, entity_type_id)