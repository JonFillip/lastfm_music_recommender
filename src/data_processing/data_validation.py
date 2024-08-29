import tensorflow_data_validation as tfdv
import pandas as pd
import yaml
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('data_validation')

def load_config():
    with open('configs/pipeline_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def generate_schema(data_path):
    try:
        log_step(logger, 'Generating Schema', 'Data Validation')
        df = pd.read_csv(data_path)
        schema = tfdv.infer_schema(df)
        tfdv.write_schema_text(schema, 'data/schema/features_schema.pbtxt')
        logger.info("Schema generated and saved successfully")
        return schema
    except Exception as e:
        log_error(logger, e, 'Schema Generation')
        raise

def validate_data(data_path, schema):
    try:
        log_step(logger, 'Validating Data', 'Data Validation')
        df = pd.read_csv(data_path)
        stats = tfdv.generate_statistics_from_dataframe(df)
        anomalies = tfdv.validate_statistics(stats, schema)
        
        if anomalies.anomaly_info:
            logger.warning("Data anomalies detected:")
            for feature, anomaly in anomalies.anomaly_info.items():
                logger.warning(f"Feature: {feature}, Anomaly: {anomaly.description}")
        else:
            logger.info("No data anomalies detected")
        
        return anomalies
    except Exception as e:
        log_error(logger, e, 'Data Validation')
        raise

def main(data_path):
    try:
        config = load_config()
        schema_path = config['data_validation']['schema_path']
        
        # Generate schema if it doesn't exist
        try:
            schema = tfdv.load_schema_text(schema_path)
        except:
            schema = generate_schema(data_path)
        
        # Validate data
        anomalies = validate_data(data_path, schema)
        
        return anomalies
    except Exception as e:
        log_error(logger, e, 'Data Validation Main')
        raise

if __name__ == '__main__':
    # This would be replaced by Kubeflow pipeline inputs
    main('data/raw/top_tracks.csv')