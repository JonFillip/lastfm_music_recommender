import yaml
import os
from google.cloud import storage
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('deploy_utils')

def load_config(config_path='configs/pipeline_config.yaml'):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        log_error(logger, e, 'Config Loading')
        raise

def upload_model_to_gcs(local_model_path, bucket_name, gcs_model_path):
    """Upload a local model to Google Cloud Storage."""
    try:
        log_step(logger, 'Uploading model to GCS', 'Deployment')
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_model_path)
        blob.upload_from_filename(local_model_path)
        logger.info(f"Model uploaded to gs://{bucket_name}/{gcs_model_path}")
    except Exception as e:
        log_error(logger, e, 'Model Upload to GCS')
        raise

def get_latest_model_version(model_dir):
    """Get the latest model version from a directory of model versions."""
    try:
        versions = [int(d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and d.isdigit()]
        return str(max(versions)) if versions else None
    except Exception as e:
        log_error(logger, e, 'Get Latest Model Version')
        raise

def validate_deployment_config(config, platform):
    """Validate the deployment configuration for a specific platform."""
    required_fields = {
        'kubernetes': ['namespace', 'deployment_name'],
        'vertex_ai': ['project_id', 'region', 'endpoint_name']
    }
    
    if platform not in required_fields:
        raise ValueError(f"Unsupported platform: {platform}")
    
    for field in required_fields[platform]:
        if field not in config or not config[field]:
            raise ValueError(f"Missing required field for {platform} deployment: {field}")

def get_model_metrics(metrics_file):
    """Load and return model metrics from a file."""
    try:
        with open(metrics_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        log_error(logger, e, 'Get Model Metrics')
        raise