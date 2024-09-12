from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
)
from src.evaluation.model_evaluation import main as evaluate_main
from src.utils.logging_utils import get_logger

logger = get_logger('kubeflow_evaluation')

@component(
    packages_to_install=['tensorflow', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn'],
    base_image='python:3.9'
)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    item_popularity: Input[Dataset],
    evaluation_results: Output[Metrics],
    evaluation_plots: Output[Dataset]
) -> float:
    import json
    import os
    
    try:
        # Create a temporary output directory
        output_dir = "/tmp/evaluation_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run evaluation
        results = evaluate_main(model.path, test_data.path, output_dir)
        
        # Save evaluation results
        with open(evaluation_results.path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Copy evaluation plots
        os.system(f"cp {output_dir}/*.png {evaluation_plots.path}")
        
        logger.info(f"Evaluation results saved to {evaluation_results.path}")
        logger.info(f"Evaluation plots saved to {evaluation_plots.path}")
        
        # Return the main model's MAP score for pipeline orchestration
        return results['main_evaluation']['mean_average_precision']
    
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model component for Kubeflow')
    parser.add_argument('--model', type=str, help='Path to the trained model')
    parser.add_argument('--test_data', type=str, help='Path to test dataset')
    parser.add_argument('--item_popularity', type=str, help='Path to item popularity data')
    parser.add_argument('--evaluation_results', type=str, help='Path to save evaluation results')
    parser.add_argument('--evaluation_plots', type=str, help='Path to save evaluation plots')
    
    args = parser.parse_args()
    
    evaluate_model(
        model=args.model,
        test_data=args.test_data,
        item_popularity=args.item_popularity,
        evaluation_results=args.evaluation_results,
        evaluation_plots=args.evaluation_plots
    )