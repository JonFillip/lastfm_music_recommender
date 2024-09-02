from ast import Tuple
import tensorflow_data_validation as tfdv
import pandas as pd
import yaml
from typing import Dict, Any, Optional
from src.utils.logging_utils import setup_logger, log_error, log_step
import os
import matplotlib.pyplot as plt
from google.cloud import storage
from datetime import datetime

logger = setup_logger('data_validation')

def load_config() -> Dict[str, Any]:
    with open('configs/pipeline_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def save_schema_to_gcs(schema: tfdv.types.Schema, bucket_name: str, model_name: str, version: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f'{model_name}/schema/schema_v{version}.pbtxt')
    blob.upload_from_string(tfdv.write_schema_text(schema))
    logger.info(f"Schema saved to gs://{bucket_name}/{model_name}/schema/schema_v{version}.pbtxt")

def load_schema_from_gcs(bucket_name: str, model_name: str, version: str) -> tfdv.types.Schema:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f'{model_name}/schema/schema_v{version}.pbtxt')
    return tfdv.load_schema_text(blob.download_as_text())

def save_statistics_to_gcs(stats: tfdv.types.DatasetFeatureStatisticsList, bucket_name: str, model_name: str, data_type: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    date = datetime.now().strftime("%Y%m%d")
    blob = bucket.blob(f'{model_name}/statistics/{data_type}_stats_{date}.pb')
    blob.upload_from_string(stats.SerializeToString())
    logger.info(f"Statistics saved to gs://{bucket_name}/{model_name}/statistics/{data_type}_stats_{date}.pb")

def load_statistics_from_gcs(bucket_name: str, model_name: str, data_type: str, date: str) -> tfdv.types.DatasetFeatureStatisticsList:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f'{model_name}/statistics/{data_type}_stats_{date}.pb')
    stats = tfdv.DatasetFeatureStatisticsList()
    stats.ParseFromString(blob.download_as_string())
    return stats

def generate_schema(data_path: str, bucket_name: str, model_name: str, version: str) -> tfdv.types.Schema:
    try:
        log_step(logger, 'Generating Schema', 'Data Validation')
        df = pd.read_csv(data_path)
        schema = tfdv.infer_schema(df)
        save_schema_to_gcs(schema, bucket_name, model_name, version)
        return schema
    except Exception as e:
        log_error(logger, e, 'Schema Generation')
        raise

def validate_data(data_path: str, schema: tfdv.types.Schema, bucket_name: str, model_name: str, data_type: str) -> Tuple[tfdv.types.DatasetFeatureStatisticsList, tfdv.types.Anomalies]:
    try:
        log_step(logger, 'Validating Data', 'Data Validation')
        df = pd.read_csv(data_path)
        stats = tfdv.generate_statistics_from_dataframe(df)
        save_statistics_to_gcs(stats, bucket_name, model_name, data_type)
        anomalies = tfdv.validate_statistics(stats, schema)
        
        if anomalies.anomaly_info:
            logger.warning("Data anomalies detected:")
            for feature, anomaly in anomalies.anomaly_info.items():
                logger.warning(f"Feature: {feature}, Anomaly: {anomaly.description}")
        else:
            logger.info("No data anomalies detected")
        
        return stats, anomalies
    except Exception as e:
        log_error(logger, e, 'Data Validation')
        raise

def compare_statistics(train_stats: tfdv.types.DatasetFeatureStatisticsList, 
                    serving_stats: tfdv.types.DatasetFeatureStatisticsList, 
                    schema: tfdv.types.Schema) -> tfdv.types.Anomalies:
    try:
        log_step(logger, 'Comparing Statistics', 'Data Validation')
        anomalies = tfdv.validate_statistics(serving_stats, schema, previous_statistics=train_stats)
        if anomalies.anomaly_info:
            logger.warning("Anomalies detected between training and serving data:")
            for feature, anomaly in anomalies.anomaly_info.items():
                logger.warning(f"Feature: {feature}, Anomaly: {anomaly.description}")
        else:
            logger.info("No anomalies detected between training and serving data")
        return anomalies
    except Exception as e:
        log_error(logger, e, 'Statistics Comparison')
        raise

def visualize_statistics(stats: tfdv.types.DatasetFeatureStatisticsList, 
                        anomalies: Optional[tfdv.types.Anomalies] = None):
    try:
        log_step(logger, 'Visualizing Statistics', 'Data Validation')
        fig = tfdv.visualize_statistics(stats, anomalies)
        plt.savefig('data/visualizations/data_statistics.png')
        logger.info("Statistics visualization saved to data/visualizations/data_statistics.png")
    except Exception as e:
        log_error(logger, e, 'Statistics Visualization')
        raise

def detect_data_drift(train_stats: tfdv.types.DatasetFeatureStatisticsList, 
                    serving_stats: tfdv.types.DatasetFeatureStatisticsList, 
                    schema: tfdv.types.Schema,
                    drift_threshold: float) -> Dict[str, float]:
    try:
        log_step(logger, 'Detecting Data Drift', 'Data Validation')
        drift_skew = tfdv.compute_drift_skew(train_stats, serving_stats, schema)
        logger.info("Data drift detected:")
        for feature, skew in drift_skew.items():
            logger.info(f"Feature: {feature}, Drift Skew: {skew}")
            if skew > drift_threshold:
                logger.warning(f"Drift threshold exceeded for feature {feature}: {skew} > {drift_threshold}")
        return drift_skew
    except Exception as e:
        log_error(logger, e, 'Data Drift Detection')
        raise

def main(train_data_path: str, serving_data_path: str, bucket_name: str, model_name: str):
    try:
        config = load_config()
        schema_version = config['data_validation']['schema_version']
        drift_threshold = config['data_validation'].get('drift_threshold', 0.1)  # Default to 0.1 if not specified
        
        # Generate or load schema
        try:
            schema = load_schema_from_gcs(bucket_name, model_name, schema_version)
            logger.info(f"Loaded existing schema version {schema_version} from GCS")
        except:
            schema = generate_schema(train_data_path, bucket_name, model_name, schema_version)
        
        # Validate training data
        train_stats, train_anomalies = validate_data(train_data_path, schema, bucket_name, model_name, 'train')
        visualize_statistics(train_stats, train_anomalies)
        
        # Validate serving data
        serving_stats, serving_anomalies = validate_data(serving_data_path, schema, bucket_name, model_name, 'serving')
        visualize_statistics(serving_stats, serving_anomalies)
        
        # Compare statistics and detect drift
        comparison_anomalies = compare_statistics(train_stats, serving_stats, schema)
        drift_skew = detect_data_drift(train_stats, serving_stats, schema, drift_threshold)
        
        return train_anomalies, serving_anomalies, comparison_anomalies, drift_skew
    except Exception as e:
        log_error(logger, e, 'Data Validation Main')
        raise

if __name__ == '__main__':
    config = load_config()
    bucket_name = config['storage']['bucket_name']
    model_name = config['model']['name']
    # Replace arg 1 and 2 with Kubeflow pipeline inputs
    main('data/raw/train_data.csv', 'data/raw/serving_data.csv', bucket_name, model_name)