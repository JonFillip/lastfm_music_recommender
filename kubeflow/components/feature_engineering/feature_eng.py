from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Artifact
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
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('kubeflow_feature_engineering')

# Define the OutputSpec NamedTuple
OutputSpec = NamedTuple('Outputs', [
    ('num_features', int),
    ('explained_variance_ratio', float),
    ('top_features', str)
])

@component(
    packages_to_install=['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'plotly'],
    base_image='python:3.10'
)
def feature_engineering(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    output_preprocessor: Output[Model],
    feature_importance_plot: Output[Artifact],
    n_components: int = 4000,
    dim_reduction_method: str = 'pca',
    feature_selection_threshold: float = 0.01
) -> OutputSpec:
    import pandas as pd
    import numpy as np
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestRegressor
    
    try:
        log_step(logger, "Loading input data", "Feature Engineering")
        df = pd.read_csv(input_data.path)
        
        log_step(logger, "Applying feature engineering steps", "Feature Engineering")
        df = engineer_basic_features(df)
        df = engineer_additional_features(df)
        df = add_tag_popularity(df)
        df = add_similar_tracks_avg_playcount(df)
        df = add_interaction_features(df)
        df = add_target_encoding(df)
        df = refine_features_further(df)
        
        log_step(logger, "Vectorizing text features", "Feature Engineering")
        df_vectorized, vectorizers = vectorize_all_text_features(df)
        
        log_step(logger, "Getting final features", "Feature Engineering")
        final_features = get_final_features(df_vectorized)
        
        log_step(logger, "Creating preprocessing pipeline", "Feature Engineering")
        preprocessor = create_preprocessing_pipeline(final_features, n_components, dim_reduction_method)
        
        log_step(logger, "Fitting preprocessor and transforming data", "Feature Engineering")
        preprocessed_data = preprocessor.fit_transform(final_features)
        
        log_step(logger, "Analyzing feature importance and reducing dimensions", "Feature Engineering")
        df_reduced, reducer, feature_importance_df = analyze_feature_importance_and_reduce_dimensions(
            pd.DataFrame(preprocessed_data, columns=preprocessor.get_feature_names_out()),
            n_components,
            dim_reduction_method
        )
        
        log_step(logger, "Performing feature selection", "Feature Engineering")
        selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), 
                                   threshold=feature_selection_threshold, prefit=False)
        selector.fit(df_reduced, df['playcount'])
        selected_features = df_reduced.columns[selector.get_support()].tolist()
        df_selected = df_reduced[selected_features]
        
        log_step(logger, "Saving preprocessed data", "Feature Engineering")
        df_selected.to_csv(output_data.path, index=False)
        logger.info(f"Preprocessed data saved to {output_data.path}")
        
        log_step(logger, "Saving preprocessor", "Feature Engineering")
        joblib.dump(preprocessor, output_preprocessor.path)
        logger.info(f"Preprocessor saved to {output_preprocessor.path}")
        
        log_step(logger, "Creating feature importance plot", "Feature Engineering")
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(feature_importance_plot.path)
        logger.info(f"Feature importance plot saved to {feature_importance_plot.path}")
        
        explained_variance_ratio = np.sum(reducer.explained_variance_ratio_) if hasattr(reducer, 'explained_variance_ratio_') else None
        top_features = ', '.join(selected_features[:10])  # Get top 10 selected features
        
        return OutputSpec(
            num_features=df_selected.shape[1],
            explained_variance_ratio=explained_variance_ratio,
            top_features=top_features
        )
    
    except Exception as e:
        log_error(logger, e, 'Feature Engineering')
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature engineering component for Kubeflow')
    parser.add_argument('--input_data', type=str, help='Path to input dataset')
    parser.add_argument('--output_data', type=str, help='Path to save the output dataset')
    parser.add_argument('--output_preprocessor', type=str, help='Path to save the preprocessor')
    parser.add_argument('--feature_importance_plot', type=str, help='Path to save the feature importance plot')
    parser.add_argument('--n_components', type=int, default=4000, help='Number of components for dimensionality reduction')
    parser.add_argument('--dim_reduction_method', type=str, default='pca', help='Dimensionality reduction method (pca, truncated_svd)')
    parser.add_argument('--feature_selection_threshold', type=float, default=0.01, help='Threshold for feature selection')
    
    args = parser.parse_args()
    
    feature_engineering(
        input_data=args.input_data,
        output_data=args.output_data,
        output_preprocessor=args.output_preprocessor,
        feature_importance_plot=args.feature_importance_plot,
        n_components=args.n_components,
        dim_reduction_method=args.dim_reduction_method,
        feature_selection_threshold=args.feature_selection_threshold
    )