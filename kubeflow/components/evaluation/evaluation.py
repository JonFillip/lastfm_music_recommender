from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
)
from typing import NamedTuple
from src.evaluation.model_evaluation import main as evaluate_main
from src.utils.logging_utils import setup_logger, log_error, log_step
from google.cloud import aiplatform

logger = setup_logger('kubeflow_evaluation')

EvaluationOutput = NamedTuple('EvaluationOutput', [
    ('mean_average_precision', float),
    ('ndcg_score', float),
    ('model_drift', float),
    ('deploy_decision', str)
])

@component(
    packages_to_install=['tensorflow', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'google-cloud-aiplatform'],
    base_image='python:3.10'
)
def evaluate_model(
    project_id: str,
    model: Input[Model],
    test_data: Input[Dataset],
    item_popularity: Input[Dataset],
    endpoint_name: str,
    evaluation_results: Output[Metrics],
    evaluation_plots: Output[Dataset],
    region: str = 'us-central1',
    map_threshold: float = 0.7,
    ndcg_threshold: float = 0.5,
    drift_threshold: float = 0.1
) -> EvaluationOutput:
    import json
    import os
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    
    try:
        log_step(logger, "Initializing evaluation", "Model Evaluation")
        
        # Create a temporary output directory
        output_dir = "/tmp/evaluation_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run evaluation on the new model
        log_step(logger, "Evaluating new model", "Model Evaluation")
        new_model_results = evaluate_main(model.path, test_data.path, output_dir)
        
        # Evaluate the currently deployed model (if exists)
        log_step(logger, "Evaluating currently deployed model", "Model Evaluation")
        aiplatform.init(project=project_id, location=region)
        endpoint = aiplatform.Endpoint(endpoint_name)
        
        if endpoint.list_models():
            current_model = endpoint.list_models()[0]
            current_model_results = evaluate_main(current_model.uri, test_data.path, output_dir)
        else:
            current_model_results = None
        
        # Calculate model drift
        if current_model_results:
            log_step(logger, "Calculating model drift", "Model Evaluation")
            new_predictions = new_model_results['predictions']
            current_predictions = current_model_results['predictions']
            model_drift = mean_absolute_error(new_predictions, current_predictions)
        else:
            model_drift = 0.0
        
        # Prepare evaluation results
        evaluation_dict = {
            'new_model': new_model_results,
            'current_model': current_model_results,
            'model_drift': model_drift
        }
        
        # Save evaluation results
        with open(evaluation_results.path, 'w') as f:
            json.dump(evaluation_dict, f, indent=2)
        
        # Copy evaluation plots
        os.system(f"cp {output_dir}/*.png {evaluation_plots.path}")
        
        logger.info(f"Evaluation results saved to {evaluation_results.path}")
        logger.info(f"Evaluation plots saved to {evaluation_plots.path}")
        
        # Make deployment decision
        new_map = new_model_results['main_evaluation']['mean_average_precision']
        new_ndcg = new_model_results['main_evaluation']['ndcg_score']
        
        if (new_map >= map_threshold and 
            new_ndcg >= ndcg_threshold and 
            model_drift <= drift_threshold):
            deploy_decision = "deploy"
        else:
            deploy_decision = "do_not_deploy"
        
        return EvaluationOutput(
            mean_average_precision=new_map,
            ndcg_score=new_ndcg,
            model_drift=model_drift,
            deploy_decision=deploy_decision
        )
    
    except Exception as e:
        log_error(logger, e, 'Model Evaluation')
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model component for Kubeflow')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model', type=str, help='Path to the trained model')
    parser.add_argument('--test_data', type=str, help='Path to test dataset')
    parser.add_argument('--item_popularity', type=str, help='Path to item popularity data')
    parser.add_argument('--endpoint_name', type=str, help='Name of the Vertex AI endpoint')
    parser.add_argument('--evaluation_results', type=str, help='Path to save evaluation results')
    parser.add_argument('--evaluation_plots', type=str, help='Path to save evaluation plots')
    parser.add_argument('--region', type=str, default='us-central1', help='GCP region')
    parser.add_argument('--map_threshold', type=float, default=0.7, help='Threshold for Mean Average Precision')
    parser.add_argument('--ndcg_threshold', type=float, default=0.5, help='Threshold for NDCG score')
    parser.add_argument('--drift_threshold', type=float, default=0.1, help='Threshold for model drift')
    
    args = parser.parse_args()
    
    evaluate_model(
        project_id=args.project_id,
        model=args.model,
        test_data=args.test_data,
        item_popularity=args.item_popularity,
        endpoint_name=args.endpoint_name,
        evaluation_results=args.evaluation_results,
        evaluation_plots=args.evaluation_plots,
        region=args.region,
        map_threshold=args.map_threshold,
        ndcg_threshold=args.ndcg_threshold,
        drift_threshold=args.drift_threshold
    )