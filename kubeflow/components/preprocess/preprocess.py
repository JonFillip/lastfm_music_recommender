from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
)
from typing import NamedTuple
from src.data_processing.data_preprocess import load_data, preprocess_data, impute_data
from src.utils.logging_utils import get_logger

logger = get_logger('kubeflow_preprocess')

# Define the OutputSpec NamedTuple
OutputSpec = NamedTuple('OutputSpec', [
    ('num_samples', int),
    ('num_features', int)
])

@component(
    packages_to_install=['pandas', 'numpy', 'scikit-learn', 'scipy'],
    base_image='python:3.9'
)
def preprocess(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
) -> OutputSpec:
    import pandas as pd
    
    try:
        # Load data
        df = load_data(input_data.path)
        
        # Preprocess data
        df_processed = preprocess_data(df)

        # Impute data for missing values
        imputed_data = impute_data(df_processed)
        imputed_data.drop_duplicates(inplace=True)

        # Save preprocessed data
        imputed_data.to_csv(output_data.path, index=False)
        logger.info(f"Preprocessed data saved to {output_data.path}")
        
        # Return number of samples and features
        return OutputSpec(num_samples=len(imputed_data), num_features=len(imputed_data.columns))
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess component for Kubeflow')
    parser.add_argument('--input_data', type=str, help='Path to input dataset')
    parser.add_argument('--output_data', type=str, help='Path to save the output dataset')
    
    args = parser.parse_args()
    
    preprocess(input_data=args.input_data, output_data=args.output_data)