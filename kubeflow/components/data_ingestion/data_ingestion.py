from kfp.v2.dsl import (
    component,
    Output,
    Dataset,
)
from typing import NamedTuple
from src.data_processing.data_ingestion import configure_lastfm_api, fetch_lastfm_data
from src.utils.logging_utils import get_logger
import os

logger = get_logger('kubeflow_data_ingestion')

# Define the OutputSpec NamedTuple
OutputSpec = NamedTuple('OutputSpec', [('num_tracks', int), ('data_version', str)])

@component(
    packages_to_install=['pylast', 'python-dotenv', 'pandas', 'requests', 'google-cloud-storage'],
    base_image='python:3.10'
)
def data_ingestion(
    project_id: str,
    output_path: Output[Dataset],
    limit: int = 5000,
) -> OutputSpec:
    import pandas as pd
    from google.cloud import storage
    from datetime import datetime
    
    try:
        # Configure GCS client
        storage_client = storage.Client(project=project_id)
        
        # Generate a unique data version
        data_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configure Last.fm API
        api_key, _ = configure_lastfm_api()
        
        # Fetch data from Last.fm
        df = fetch_lastfm_data(api_key, limit=limit)
        
        if not df.empty:
            logger.info(f"Successfully fetched {len(df)} tracks from Last.fm")
            
            # Save data to GCS
            bucket_name, blob_name = output_path.path.replace("gs://", "").split("/", 1)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(f"{blob_name}_{data_version}.csv")
            
            blob.upload_from_string(df.to_csv(index=False), content_type='text/csv')
            logger.info(f"Data saved to {output_path.path}_{data_version}.csv")
            
            return OutputSpec(num_tracks=len(df), data_version=data_version)
        else:
            logger.error("Failed to fetch data, DataFrame is empty")
            return OutputSpec(num_tracks=0, data_version=data_version)
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data ingestion component for Kubeflow')
    parser.add_argument('--project_id', type=str, required=True, help='GCP Project ID')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output dataset')
    parser.add_argument('--limit', type=int, default=5000, help='Number of tracks to fetch')
    
    args = parser.parse_args()
    
    data_ingestion(project_id=args.project_id, output_path=args.output_path, limit=args.limit)