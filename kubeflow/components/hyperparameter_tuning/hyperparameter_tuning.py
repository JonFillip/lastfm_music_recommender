from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
)
from typing import NamedTuple
from src.hyperparameter_tuning.katib_tuning import run_hyperparameter_tuning
from src.utils.logging_utils import setup_logger, log_error

logger = setup_logger('kubeflow_hyperparameter_tuning')

OutputSpec = NamedTuple('Outputs', [('best_val_cosine_similarity', float)])

@component(
    packages_to_install=['kubeflow-katib', 'PyYAML'],
    base_image='python:3.9'
)
def hyperparameter_tuning(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    best_hyperparameters: Output[Dataset]
) -> OutputSpec:
    import json
    
    try:
        # Run hyperparameter tuning
        results = run_hyperparameter_tuning(train_data.path, val_data.path)
        
        # Extract best hyperparameters and performance
        best_params = {param['name']: param['value'] for param in results['currentOptimalTrial']['parameterAssignments']}
        best_metric = next(metric for metric in results['currentOptimalTrial']['observation']['metrics'] if metric['name'] == 'val_cosine_similarity')
        best_val_cosine_similarity = float(best_metric['value'])
        
        # Save best hyperparameters
        with open(best_hyperparameters.path, 'w') as f:
            json.dump(best_params, f)
        
        logger.info(f"Best hyperparameters saved to {best_hyperparameters.path}")
        logger.info(f"Best validation cosine similarity: {best_val_cosine_similarity}")
        
        return (best_val_cosine_similarity,)
    
    except Exception as e:
        log_error(logger, e, 'Hyperparameter Tuning')
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning component for Kubeflow')
    parser.add_argument('--train_data', type=str, help='Path to training dataset')
    parser.add_argument('--val_data', type=str, help='Path to validation dataset')
    parser.add_argument('--best_hyperparameters', type=str, help='Path to save the best hyperparameters')
    
    args = parser.parse_args()
    
    hyperparameter_tuning(
        train_data=args.train_data,
        val_data=args.val_data,
        best_hyperparameters=args.best_hyperparameters
    )