from kfp.v2.dsl import (
    component,
    Output,
    Dataset,
)
from typing import NamedTuple
from src.data_processing.data_ingestion import configure_lastfm_api, fetch_lastfm_data
from src.utils.logging_utils import get_logger

logger = get_logger('kubeflow_data_ingestion')

# Define the OutputSpec NamedTuple
OutputSpec = NamedTuple('OutputSpec', [('num_tracks', int)])

@component(
    packages_to_install=['pylast', 'python-dotenv', 'pandas', 'requests'],
    base_image='python:3.9'
)
def data_ingestion(
    output_path: Output[Dataset],
    limit: int = 5000,
) -> OutputSpec:
    import os
    import pandas as pd
    
    try:
        api_key, _ = configure_lastfm_api()
        df = fetch_lastfm_data(api_key, limit=limit)
        
        if not df.empty:
            logger.info(f"Successfully fetched {len(df)} tracks from Last.fm")
            df.to_csv(output_path.path, index=False)
            logger.info(f"Data saved to {output_path.path}")
            return (len(df),)
        else:
            logger.error("Failed to fetch data, DataFrame is empty")
            return (0,)
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data ingestion component for Kubeflow')
    parser.add_argument('--output_path', type=str, help='Path to save the output dataset')
    parser.add_argument('--limit', type=int, default=5000, help='Number of tracks to fetch')
    
    args = parser.parse_args()
    
    data_ingestion(output_path=args.output_path, limit=args.limit)