import argparse
import tensorflow as tf
import numpy as np
import json
from src.utils.logging_utils import setup_logger, log_error, log_metric, log_step
import yaml

logger = setup_logger('model_evaluation')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, X_test, y_test, config):
    try:
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

def main(model_path, test_data_path, config_path, output_path):
    try:
        config = load_config(config_path)
        model = tf.keras.models.load_model(model_path)
        test_data = np.load(test_data_path)
        X_test, y_test = test_data['X'], test_data['y']
        
        evaluation_results = evaluate_model(model, X_test, y_test, config)
        
        # Save results to file
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f)
        
        logger.info(f"Evaluation results saved to {output_path}")
        return evaluation_results
    except Exception as e:
        log_error(logger, e, 'Model Evaluation Main')
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the music recommender model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the evaluation results')
    
    args = parser.parse_args()
    
    main(args.model_path, args.test_data_path, args.config_path, args.output_path)