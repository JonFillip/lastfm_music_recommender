import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.sparse import csr_matrix, hstack
import multiprocessing as mp
from functools import partial
from src.utils.logging_utils import setup_logger

logger = setup_logger('feature_engineering')

def load_data_from_bigquery(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    """
    return client.query(query).to_dataframe()

def write_to_bigquery(df, project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(autodetect=True, write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, f"{project_id}.{dataset_id}.{table_id}", job_config=job_config)
    job.result()

def engineer_basic_features(df):
    df['log_playcount'] = np.log1p(df['playcount'])
    df['num_tags'] = df['tags'].str.count(',') + 1
    df['num_similar_tracks'] = df['similar_tracks'].str.count(',') + 1
    return df

def engineer_additional_features(df):
    df['high_tag_count'] = (df['num_tags'] > df['num_tags'].median()).astype(int)
    df['has_similar_tracks'] = (df['num_similar_tracks'] > 0).astype(int)
    df['log_playcount_x_num_tags'] = df['log_playcount'] * df['num_tags']
    return df

def add_tag_popularity(df):
    tag_df = df['tags'].str.split(',', expand=True).melt(value_name='tag').dropna()
    tag_df = tag_df.merge(df[['log_playcount']], left_index=True, right_index=True)
    tag_popularity = tag_df.groupby('tag')['log_playcount'].mean().sort_values(ascending=False)
    df['avg_tag_popularity'] = df['tags'].apply(lambda x: tag_popularity[x.split(',')].mean() if x else 0)
    return df

def add_similar_tracks_avg_playcount(df):
    track_playcount = dict(zip(df['name'], df['log_playcount']))
    df['avg_similar_tracks_playcount'] = df['similar_tracks'].apply(
        lambda x: np.mean([track_playcount.get(t.strip(), 0) for t in x.split(',')]) if x else 0
    )
    return df

def vectorize_text_features(df, max_features_dict):
    vectorizers = {}
    for feature, max_features in max_features_dict.items():
        vectorizer = TfidfVectorizer(max_features=max_features)
        vectorized = vectorizer.fit_transform(df[feature].fillna(''))
        df = pd.concat([df, pd.DataFrame(vectorized.toarray(), columns=[f'{feature}_tfidf_{i}' for i in range(vectorized.shape[1])], index=df.index)], axis=1)
        vectorizers[feature] = vectorizer
    return df, vectorizers

def feature_selection(X, y, k=1000):
    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_new, selected_features

def process_chunk(chunk, feature_engineering_pipeline):
    return feature_engineering_pipeline(chunk)

def feature_engineering_pipeline(df):
    df = engineer_basic_features(df)
    df = engineer_additional_features(df)
    df = add_tag_popularity(df)
    df = add_similar_tracks_avg_playcount(df)
    df, _ = vectorize_text_features(df, {'tags': 300, 'similar_tracks': 500})
    return df

def main(project_id, input_dataset_id, input_table_id, output_dataset_id, output_table_id):
    try:
        logger.info("Starting feature engineering process")
        df = load_data_from_bigquery(project_id, input_dataset_id, input_table_id)
        
        # Process data in chunks to optimize memory usage
        chunk_size = 10000
        with mp.Pool(mp.cpu_count()) as pool:
            processed_chunks = pool.map(
                partial(process_chunk, feature_engineering_pipeline=feature_engineering_pipeline),
                [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            )
        
        df_processed = pd.concat(processed_chunks)
        
        # Perform feature selection
        X = df_processed.drop(['name', 'artist', 'tags', 'similar_tracks', 'playcount'], axis=1)
        y = df_processed['log_playcount']
        X_selected, selected_features = feature_selection(X, y)
        
        df_final = pd.concat([df_processed[['name', 'artist', 'tags', 'similar_tracks', 'playcount']], pd.DataFrame(X_selected, columns=selected_features, index=df_processed.index)], axis=1)
        
        write_to_bigquery(df_final, project_id, output_dataset_id, output_table_id)
        logger.info("Feature engineering completed successfully")
    
    except Exception as e:
        logger.error(f"Error in feature engineering process: {e}")
        raise

if __name__ == "__main__":
    project_id = "your-project-id"
    input_dataset_id = "lastfm_dataset"
    input_table_id = "processed_top_tracks"
    output_dataset_id = "lastfm_dataset"
    output_table_id = "engineered_top_tracks"
    main(project_id, input_dataset_id, input_table_id, output_dataset_id, output_table_id)
