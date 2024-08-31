import argparse
import os
from src.data_processing.data_ingestion import fetch_lastfm_data, configure_lastfm_api
from src.utils.logging_utils import get_logger

logger = get_logger('data_ingestion_component')

def main(api_key: str, limit: int, output_csv: str):
    try:
        df = fetch_lastfm_data(api_key, limit=limit)
        if not df.empty:
            logger.info(f"Successfully fetched {len(df)} tracks from Last.fm")
            df.to_csv(output_csv, index=False)
            logger.info(f"Data saved to {output_csv}")
        else:
            logger.error("Failed to fetch data, DataFrame is empty")
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Ingestion Component')
    parser.add_argument('--api_key', type=str, required=True, help='Last.fm API key')
    parser.add_argument('--limit', type=str, default='5000', help='Number of tracks to fetch')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV')
    
    args = parser.parse_args()
    limit = int(args.limit)
    
    main(args.api_key, limit, args.output_csv)