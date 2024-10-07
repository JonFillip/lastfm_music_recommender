from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Artifact
)
from typing import NamedTuple
from src.hyperparameter_tuning.katib_tuning import run_hyperparameter_tuning
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('kubeflow_hyperparameter_tuning')

OutputSpec = NamedTuple('Outputs', [
    ('best_val_cosine_similarity', float),
    ('best_val_ndcg', float)
])

@component(
    packages_to_install=['kubeflow-katib', 'PyYAML', 'matplotlib', 'seaborn'],
    base_image='python:3.10'
)
def hyperparameter_tuning(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    best_hyperparameters: Output[Dataset],
    tuning_results_plot: Output[Artifact],
    search_algorithm: str = 'bayesian',
    max_trials: int = 50,
    max_duration_minutes: int = 120,
    early_stopping_rounds: int = 10
) -> OutputSpec:
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    try:
        log_step(logger, "Starting hyperparameter tuning", "Hyperparameter Tuning")
        results = run_hyperparameter_tuning(
            train_data.path, 
            val_data.path, 
            search_algorithm=search_algorithm,
            max_trials=max_trials,
            max_duration_minutes=max_duration_minutes,
            early_stopping_rounds=early_stopping_rounds
        )
        
        log_step(logger, "Extracting best hyperparameters and performance", "Hyperparameter Tuning")
        best_params = {param['name']: param['value'] for param in results['currentOptimalTrial']['parameterAssignments']}
        best_metrics = {metric['name']: float(metric['value']) for metric in results['currentOptimalTrial']['observation']['metrics']}
        
        best_val_cosine_similarity = best_metrics.get('val_cosine_similarity', 0.0)
        best_val_ndcg = best_metrics.get('val_ndcg', 0.0)
        
        log_step(logger, "Saving best hyperparameters", "Hyperparameter Tuning")
        with open(best_hyperparameters.path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"Best hyperparameters saved to {best_hyperparameters.path}")
        logger.info(f"Best validation cosine similarity: {best_val_cosine_similarity}")
        logger.info(f"Best validation NDCG: {best_val_ndcg}")
        
        log_step(logger, "Visualizing hyperparameter tuning results", "Hyperparameter Tuning")
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            x=[trial['observation']['metrics'][0]['value'] for trial in results['trials']],
            y=[trial['observation']['metrics'][1]['value'] for trial in results['trials']]
        )
        plt.xlabel('Validation Cosine Similarity')
        plt.ylabel('Validation NDCG')
        plt.title('Hyperparameter Tuning Results')
        plt.savefig(tuning_results_plot.path)
        logger.info(f"Tuning results plot saved to {tuning_results_plot.path}")
        
        return OutputSpec(
            best_val_cosine_similarity=best_val_cosine_similarity,
            best_val_ndcg=best_val_ndcg
        )
    
    except Exception as e:
        log_error(logger, e, 'Hyperparameter Tuning')
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning component for Kubeflow')
    parser.add_argument('--train_data', type=str, help='Path to training dataset')
    parser.add_argument('--val_data', type=str, help='Path to validation dataset')
    parser.add_argument('--best_hyperparameters', type=str, help='Path to save the best hyperparameters')
    parser.add_argument('--tuning_results_plot', type=str, help='Path to save the tuning results plot')
    parser.add_argument('--search_algorithm', type=str, default='bayesian', help='Search algorithm for hyperparameter tuning')
    parser.add_argument('--max_trials', type=int, default=50, help='Maximum number of trials for hyperparameter tuning')
    parser.add_argument('--max_duration_minutes', type=int, default=120, help='Maximum duration for hyperparameter tuning in minutes')
    parser.add_argument('--early_stopping_rounds', type=int, default=10, help='Number of rounds for early stopping')
    
    args = parser.parse_args()
    
    hyperparameter_tuning(
        train_data=args.train_data,
        val_data=args.val_data,
        best_hyperparameters=args.best_hyperparameters,
        tuning_results_plot=args.tuning_results_plot,
        search_algorithm=args.search_algorithm,
        max_trials=args.max_trials,
        max_duration_minutes=args.max_duration_minutes,
        early_stopping_rounds=args.early_stopping_rounds
    )