import argparse
import logging
from typing import List, Dict
from google.cloud import aiplatform
from google.cloud.aiplatform import FeatureStore
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_features(df: pd.DataFrame, max_features_per_group: int = 1000) -> List[pd.DataFrame]:
    """
    Split the input DataFrame into multiple DataFrames, each with at most max_features_per_group columns.
    """
    feature_groups = []
    for i in range(0, df.shape[1], max_features_per_group):
        feature_groups.append(df.iloc[:, i:i+max_features_per_group])
    return feature_groups

def create_and_populate_feature_store(
    project_id: str,
    region: str,
    feature_store_id: str,
    entity_type_id_prefix: str,
    input_data: str,
    feature_store_uri: str
) -> None:
    try:
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)

        # Load the input data
        df = pd.read_csv(input_data)
        
        # Ensure there's an 'entity_id' column, if not, create one
        if 'entity_id' not in df.columns:
            df['entity_id'] = df.index.astype(str)

        # Split features into groups
        feature_groups = split_features(df.drop('entity_id', axis=1))

        # Create a feature store
        fs = FeatureStore.create(
            feature_store_id=feature_store_id,
            online_store_fixed_node_count=1,
            sync=True
        )
        logger.info(f"Created Feature Store: {fs.name}")

        # Create entity types and ingest features for each group
        for i, feature_group in enumerate(feature_groups):
            entity_type_id = f"{entity_type_id_prefix}_{i+1}"
            
            # Create an entity type
            entity_type = fs.create_entity_type(
                entity_type_id=entity_type_id,
                description=f"Music track features group {i+1}"
            )
            logger.info(f"Created Entity Type: {entity_type.name}")

            # Create features
            for feature_id in feature_group.columns:
                feature_type = "DOUBLE" if np.issubdtype(feature_group[feature_id].dtype, np.number) else "STRING"
                entity_type.create_feature(
                    feature_id=feature_id,
                    value_type=feature_type,
                    description=f"Feature: {feature_id}"
                )
                logger.info(f"Created feature: {feature_id}")

            # Prepare data for ingestion
            ingestion_data = pd.concat([df['entity_id'], feature_group], axis=1)
            ingestion_data['timestamp'] = pd.Timestamp.now()

            # Ingest feature values
            entity_type.ingest(
                source=ingestion_data.to_dict('records'),
                entity_id_field="entity_id",
                feature_time_field="timestamp"
            )
            logger.info(f"Ingested feature values for group {i+1}")

        # Write the feature store URI to the output file
        with open(feature_store_uri, 'w') as f:
            f.write(fs.name)
        logger.info(f"Feature Store URI written to: {feature_store_uri}")

    except Exception as e:
        logger.error(f"Error in create_and_populate_feature_store: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create and populate Vertex AI Feature Store')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--region', required=True, help='GCP Region')
    parser.add_argument('--feature_store_id', required=True, help='Feature Store ID')
    parser.add_argument('--entity_type_id_prefix', required=True, help='Prefix for Entity Type IDs')
    parser.add_argument('--input_data', required=True, help='Path to input data CSV file')
    parser.add_argument('--feature_store_uri', required=True, help='Output path for Feature Store URI')

    args = parser.parse_args()

    create_and_populate_feature_store(
        args.project_id,
        args.region,
        args.feature_store_id,
        args.entity_type_id_prefix,
        args.input_data,
        args.feature_store_uri
    )