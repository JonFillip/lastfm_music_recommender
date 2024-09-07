import tensorflow as tf
import numpy as np
from src.utils.logging_utils import setup_logger, log_error, log_metric, log_step
import yaml

logger = setup_logger('model_evaluation')

def load_config():
    with open('configs/pipeline_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, X_test, y_test):
    try:
        config = load_config()
        log_step(logger, 'Model Evaluation', 'Evaluation')
        
        metrics = config['model_evaluation']['metrics']
        results = model.evaluate(X_test, y_test, return_dict=True)
        
        for metric in metrics:
            if metric in results:
                log_metric(logger, metric, results[metric], 'Model Evaluation')
        
        return results
    except Exception as e:
        log_error(logger, e, 'Model Evaluation')
        raise

def main(model_path, test_data_path):
    try:
        model = tf.keras.models.load_model(model_path)
        test_data = np.load(test_data_path)
        X_test, y_test = test_data['X'], test_data['y']
        
        evaluation_results = evaluate_model(model, X_test, y_test)
        return evaluation_results
    except Exception as e:
        log_error(logger, e, 'Model Evaluation Main')
        raise

if __name__ == '__main__':
    # This would be replaced by Kubeflow pipeline inputs
    main('/models/content_based_model', '/data/processed/test.npz')