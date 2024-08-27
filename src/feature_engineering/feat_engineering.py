import pandas as pd 
import numpy as np
import sklearn
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, KBinsDiscretizer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from itertools import combinations
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.data_utilis import plot_correlation_map

def engineer_basic_features(df):
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    new_df = df.copy()
    
    # Log-transform the playcount
    new_df.loc[:, 'log_playcount'] = np.log1p(new_df['playcount'])
    
    # Create binned versions of playcount
    kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    new_df.loc[:, 'binned_playcount'] = kbd.fit_transform(new_df[['playcount']])
    
    # Create features for the number of tags and similar tracks
    new_df.loc[:, 'num_tags'] = new_df['tags'].apply(lambda x: len(x.split(', ')))
    new_df.loc[:, 'num_similar_tracks'] = new_df['similar_tracks'].apply(lambda x: len(x.split(', ')))
    
    return new_df


def plot_new_features(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    sns.histplot(data=df, x='log_playcount', kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Log Playcount')
    
    sns.histplot(data=df, x='binned_playcount', kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Binned Playcount')
    
    sns.histplot(data=df, x='num_tags', kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Number of Tags')
    
    sns.histplot(data=df, x='num_similar_tracks', kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Number of Similar Tracks')
    
    plt.tight_layout()
    plt.show()

def engineer_additional_features(df):
    # Create a copy of the dataframe
    new_df = df.copy()
    
    # Binary feature for high tag count
    new_df['high_tag_count'] = (new_df['num_tags'] > 50).astype(int)
    
    # Bin number of tags
    new_df['tag_count_category'] = pd.cut(new_df['num_tags'], 
                                        bins=[0, 10, 50, np.inf], 
                                        labels=['low', 'medium', 'high'])
    
    # Binary feature for having similar tracks
    new_df['has_similar_tracks'] = (new_df['num_similar_tracks'] > 0).astype(int)
    
    # Bin number of similar tracks
    new_df['similar_tracks_category'] = pd.cut(new_df['num_similar_tracks'], 
                                            bins=[0, 50, 100, np.inf], 
                                            labels=['low', 'medium', 'high'])
    
    # Interaction features
    new_df['log_playcount_x_num_tags'] = new_df['log_playcount'] * new_df['num_tags']
    new_df['log_playcount_x_num_similar_tracks'] = new_df['log_playcount'] * new_df['num_similar_tracks']
    
    return new_df

def drop_columns(features, df):
    new_df = df.copy()
    imputed_data_with_more_features = new_df.drop(features, axis=1)

    return imputed_data_with_more_features

def correlation_map(df):
    new_df = df.copy()

    plot_correlation_map(new_df)

def refine_features(df):
    # Create a copy of the dataframe
    new_df = df.copy()
    
    # Drop redundant features
    new_df = new_df.drop(['playcount', 'binned_playcount', 'high_tag_count', 
                        'log_playcount_x_num_tags', 'log_playcount_x_num_similar_tracks'], axis=1)
    
    # Create artist-based features
    new_df['artist_avg_playcount'] = new_df.groupby('artist')['log_playcount'].transform('mean')
    new_df['artist_track_count'] = new_df.groupby('artist')['name'].transform('count')
    
    # Create features for top N tags
    top_tags = new_df['tags'].str.split(', ', expand=True).stack().value_counts().nlargest(10).index
    for tag in top_tags:
        new_df[f'has_tag_{tag}'] = new_df['tags'].str.contains(tag).astype(int)
    
    return new_df

def add_tag_popularity(df):
    # Split tags and create a dataframe
    tag_df = df['tags'].str.split(', ', expand=True).melt(value_name='tag').dropna()
    tag_df = tag_df.merge(df[['log_playcount']], left_index=True, right_index=True)
    
    # Calculate tag popularity
    tag_popularity = tag_df.groupby('tag')['log_playcount'].mean().sort_values(ascending=False)
    
    # Function to calculate average tag popularity
    def avg_tag_popularity(tags):
        if not tags:
            return 0
        tags_list = tags.split(', ')
        # Only consider tags that are in tag_popularity
        valid_tags = [tag for tag in tags_list if tag in tag_popularity.index]
        if not valid_tags:
            return 0
        return tag_popularity[valid_tags].mean()
    
    # Add tag popularity to main dataframe
    df['avg_tag_popularity'] = df['tags'].apply(avg_tag_popularity)
    
    return df

def add_similar_tracks_avg_playcount(df):
    # Create a dictionary of track name to log_playcount
    track_playcount = dict(zip(df['name'], df['log_playcount']))
    
    # Function to get average playcount of similar tracks
    def get_avg_playcount(similar_tracks):
        playcounts = [track_playcount.get(track.strip(), 0) for track in similar_tracks.split(', ')]
        return sum(playcounts) / len(playcounts) if playcounts else 0
    
    df['avg_similar_tracks_playcount'] = df['similar_tracks'].apply(get_avg_playcount)
    
    return df

def add_interaction_features(df):
    df['num_tags_x_avg_similar_tracks_playcount'] = df['num_tags'] * df['avg_similar_tracks_playcount']
    
    return df

def add_target_encoding(df):
    # Calculate mean log_playcount for each artist
    artist_means = df.groupby('artist')['log_playcount'].mean()
    
    # Calculate global mean
    global_mean = df['log_playcount'].mean()
    
    # Function to encode with smoothing
    def encode(artist):
        n = df[df['artist'] == artist].shape[0]
        return (n * artist_means.get(artist, global_mean) + global_mean) / (n + 1)
    
    # Apply encoding
    df['artist_target_encoded'] = df['artist'].apply(encode)
    
    return df

def refine_features_further(df):
    # Combine redundant features
    df['has_tag_favorites_combined'] = df[['has_tag_favorites', 'has_tag_Favorite']].max(axis=1)

    # drop low variance features
    df = df.drop(['has_tag_favorites', 'has_tag_Favorite', 'has_tag_MySpotigramBot'], axis=1)
    
    # Create a composite tag popularity score
    tag_columns = [col for col in df.columns if col.startswith('has_tag_')]
    df['tag_popularity_score'] = df[tag_columns].mean(axis=1)
    
    return df

def review_categorical_features(df):
    cat_cols = ['tag_count_category', 'similar_tracks_category']
    for col in cat_cols:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())

def analyze_vocabulary_sizes(df):
    text_features = ['name', 'artist', 'tags', 'similar_tracks']
    for feature in text_features:
        unique_terms = set()
        for text in df[feature]:
            unique_terms.update(text.split(','))
        print(f"{feature} unique terms: {len(unique_terms)}")

def remove_pretfidf_cols(df):
    # Identify and remove previously vectorized track name features
    name_tfidf_columns = [col for col in df.columns if col.startswith('name_tfidf_')]
    refined_data_new = refined_data_new.drop(columns=name_tfidf_columns)

    return refined_data_new


def vectorize_all_text_features(df, max_features_dict=None):
    if max_features_dict is None:
        max_features_dict = {
            'artist': None,  # This will use all unique artists
            'tags': 300,
            'similar_tracks': 500
        }
    
    # Check if 'name' has already been vectorized
    if 'name' not in max_features_dict and not any(col.startswith('name_tfidf_') for col in df.columns):
        max_features_dict['name'] = 8000
    
    text_features = list(max_features_dict.keys())
    vectorized_dfs = []
    vectorizers = {}

    for feature in text_features:
        if feature in ['name', 'artist']:
            # Treat each unique value as a document
            unique_values = df[feature].unique()
            text_data = pd.Series(unique_values)
        else:
            text_data = df[feature].fillna('')

        tfidf = TfidfVectorizer(max_features=max_features_dict[feature])
        tfidf_matrix = tfidf.fit_transform(text_data)
        
        feature_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'{feature}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )

        if feature in ['name', 'artist']:
            # Map the vectorized features back to the original dataframe
            feature_to_vector = dict(zip(unique_values, feature_df.values))
            vectorized_feature = df[feature].map(lambda x: feature_to_vector.get(x, np.zeros(max_features_dict[feature])))
            feature_df = pd.DataFrame(vectorized_feature.tolist(), 
                                    columns=feature_df.columns, 
                                    index=df.index)
        else:
            feature_df.index = df.index

        vectorized_dfs.append(feature_df)
        vectorizers[feature] = tfidf

    df_vectorized = pd.concat([df] + vectorized_dfs, axis=1)
    
    return df_vectorized, vectorizers


def get_final_features(df):
    # Prepare final feature set
    original_text_cols = ['name', 'artist', 'tags', 'similar_tracks']
    feature_cols = [col for col in df.columns if col not in original_text_cols]
    print("Final feature set:")
    print(feature_cols)
    return df[feature_cols]


def create_preprocessing_pipeline(df, n_components=100):
    # Identify different types of columns
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    tfidf_features = [col for col in df.columns if '_tfidf_' in col]
    
    # Create the preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    tfidf_transformer = Pipeline(steps=[
        ('svd', TruncatedSVD(n_components=min(n_components, len(tfidf_features)), algorithm='randomized', n_iter=5, random_state=42))
    ])
    
    # Combine all the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('tfidf', tfidf_transformer, tfidf_features)
        ])
    
    # Create the full pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('replace_inf', FunctionTransformer(lambda X: np.nan_to_num(X, nan=0, posinf=0, neginf=0)))
    ])
    
    return full_pipeline

def get_feature_names(input_features, n_components):
    feature_names = []
    numeric_features = input_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = input_features.select_dtypes(include=['object']).columns.tolist()
    tfidf_features = [col for col in input_features.columns if '_tfidf_' in col]
    
    feature_names.extend(numeric_features)
    
    for cat_feature in categorical_features:
        unique_values = input_features[cat_feature].unique()
        feature_names.extend([f"{cat_feature}_{value}" for value in unique_values])
    
    feature_names.extend([f'svd_tfidf_{i}' for i in range(min(n_components, len(tfidf_features)))])
    
    return feature_names

def analyze_feature_importance_and_reduce_dimensions(df, n_components=4000):
    # Perform Truncated SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_result = svd.fit_transform(df)
    
    # Analyze feature importance
    feature_importance = np.sum(np.abs(svd.components_), axis=0)
    feature_importance = 100.0 * (feature_importance / feature_importance.sum())
    
    # Create a DataFrame of feature importance
    feature_importance_df = pd.DataFrame({
        'feature': df.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(20), feature_importance_df['importance'][:20])
    plt.xticks(range(20), feature_importance_df['feature'][:20], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Relative Importance (%)')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.show()
    
    # Print top 20 most important features
    print("Top 20 most important features:")
    print(feature_importance_df.head(20))
    
    # Plot cumulative explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(svd.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Number of Components')
    plt.tight_layout()
    plt.show()
    
    # Create DataFrame with reduced dimensions
    columns = [f'SVD_{i+1}' for i in range(svd_result.shape[1])]
    df_svd = pd.DataFrame(svd_result, columns=columns, index=df.index)
    
    print(f"Shape after Truncated SVD: {df_svd.shape}")
    print(f"Cumulative explained variance ratio: {np.sum(svd.explained_variance_ratio_):.4f}")
    
    return df_svd, svd, feature_importance_df

