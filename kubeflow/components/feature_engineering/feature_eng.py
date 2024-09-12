from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model
)
from typing import NamedTuple
from src.feature_engineering.feat_engineering import (
    engineer_basic_features,
    engineer_additional_features,
    add_tag_popularity,
    add_similar_tracks_avg_playcount,
    add_interaction_features,
    add_target_encoding,
    refine_features_further,
    vectorize_all_text_features,
    get_final_features,
    create_preprocessing_pipeline,
    analyze_feature_importance_and_reduce_dimensions
)
from src.utils.logging_utils import get_logger

logger = get_logger('kubeflow_feature_engineering')

# Define the OutputSpec NamedTuple
OutputSpec = NamedTuple('Outputs', [('num_features', int),('explained_variance_ratio', float)])

@component(
    packages_to_install=['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn'],
    base_image='python:3.9'
)
def feature_engineering(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    output_preprocessor: Output[Model],
    n_components: int = 4000
) -> OutputSpec:
    import pandas as pd
    import numpy as np
    import joblib
    
    try:
        # Load data
        df = pd.read_csv(input_data.path)
        
        # Apply feature engineering steps
        df = engineer_basic_features(df)
        df = engineer_additional_features(df)
        df = add_tag_popularity(df)
        df = add_similar_tracks_avg_playcount(df)
        df = add_interaction_features(df)
        df = add_target_encoding(df)
        df = refine_features_further(df)
        
        # Vectorize text features
        df_vectorized, vectorizers = vectorize_all_text_features(df)
        
        # Get final features
        final_features = get_final_features(df_vectorized)
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(final_features, n_components)
        
        # Fit preprocessor and transform data
        preprocessed_data = preprocessor.fit_transform(final_features)
        
        # Analyze feature importance and reduce dimensions
        df_svd, svd, feature_importance_df = analyze_feature_importance_and_reduce_dimensions(
            pd.DataFrame(preprocessed_data, columns=preprocessor.get_feature_names_out()),
            n_components
        )
        
        # Save preprocessed data
        df_svd.to_csv(output_data.path, index=False)
        logger.info(f"Preprocessed data saved to {output_data.path}")
        
        # Save preprocessor
        joblib.dump(preprocessor, output_preprocessor.path)
        logger.info(f"Preprocessor saved to {output_preprocessor.path}")
        
        return (df_svd.shape[1], np.sum(svd.explained_variance_ratio_))
    
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature engineering component for Kubeflow')
    parser.add_argument('--input_data', type=str, help='Path to input dataset')
    parser.add_argument('--output_data', type=str, help='Path to save the output dataset')
    parser.add_argument('--output_preprocessor', type=str, help='Path to save the preprocessor')
    parser.add_argument('--n_components', type=int, default=4000, help='Number of components for dimensionality reduction')
    
    args = parser.parse_args()
    
    feature_engineering(
        input_data=args.input_data,
        output_data=args.output_data,
        output_preprocessor=args.output_preprocessor,
        n_components=args.n_components
    )