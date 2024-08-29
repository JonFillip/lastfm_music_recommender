from kserve import KFServingClient
from kubernetes import client
from kubernetes.client import V1Container
import yaml
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('model_serving')

def load_config():
    with open('configs/pipeline_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def deploy_model(model_path):
    try:
        config = load_config()
        log_step(logger, 'Model Deployment', 'Serving')
        
        kserve_client = KFServingClient()
        
        model_name = config['model_serving']['model_name']
        model_version = config['model_serving']['model_version']
        
        isvc = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": model_name,
                "namespace": "default"
            },
            "spec": {
                "predictor": {
                    "tensorflow": {
                        "storageUri": model_path
                    }
                }
            }
        }
        
        kserve_client.create(isvc)
        logger.info(f"Model {model_name} version {model_version} deployed successfully")
    except Exception as e:
        log_error(logger, e, 'Model Deployment')
        raise

def main(model_path):
    try:
        deploy_model(model_path)
    except Exception as e:
        log_error(logger, e, 'Model Serving Main')
        raise

if __name__ == '__main__':
    # This would be replaced by Kubeflow pipeline inputs
    main('gs://your-bucket/models/content_based_model')