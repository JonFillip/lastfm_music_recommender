from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
)
from typing import NamedTuple
from src.data_processing.data_prep import prepare_data
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('kubeflow_preprocess')

# Define the OutputSpec NamedTuple
OutputSpec = NamedTuple('OutputSpec', [
    ('num_samples', int),
    ('num_features', int),
    ('train_samples', int),
    ('val_samples', int),
    ('test_samples', int)
])

@component(
    packages_to_install=['pandas', 'numpy', 'scikit-learn'],
    base_image='python:3.10'
)
def preprocess(
    input_data: Input[Dataset],
    output_train: Output[Dataset],
    output_val: Output[Dataset],
    output_test: Output[Dataset],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> OutputSpec:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    try:
        log_step(logger, "Loading data", "Preprocessing")
        df = pd.read_csv(input_data.path)
        
        log_step(logger, "Preparing data for model training and testing", "Preprocessing")
        prepared_data = prepare_data(df)
        
        log_step(logger, "Splitting data into train, validation, and test sets", "Preprocessing")
        train_val, test = train_test_split(prepared_data, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)

        log_step(logger, "Saving preprocessed datasets", "Preprocessing")
        train.to_csv(output_train.path, index=False)
        val.to_csv(output_val.path, index=False)
        test.to_csv(output_test.path, index=False)
        logger.info(f"Train data saved to {output_train.path}")
        logger.info(f"Validation data saved to {output_val.path}")
        logger.info(f"Test data saved to {output_test.path}")

        return OutputSpec(
            num_samples=len(prepared_data),
            num_features=len(prepared_data.columns),
            train_samples=len(train),
            val_samples=len(val),
            test_samples=len(test)
        )
    
    except Exception as e:
        log_error(logger, e, 'Preprocessing')
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess component for Kubeflow')
    parser.add_argument('--input_data', type=str, help='Path to input dataset')
    parser.add_argument('--output_train', type=str, help='Path to save the training dataset')
    parser.add_argument('--output_val', type=str, help='Path to save the validation dataset')
    parser.add_argument('--output_test', type=str, help='Path to save the test dataset')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of data to use for validation')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    preprocess(
        input_data=args.input_data,
        output_train=args.output_train,
        output_val=args.output_val,
        output_test=args.output_test,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )