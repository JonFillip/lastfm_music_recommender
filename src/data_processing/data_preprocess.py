import ast
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from collections import Counter
from itertools import combinations
import gc
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
import seaborn as sns


def robust_string_parser(x):
    if pd.isna(x):
        return []  # Return empty list instead of np.nan
    if isinstance(x, str):
        return [item.strip() for item in x.split(',') if item.strip()]
    if isinstance(x, (list, np.ndarray)):
        return [str(item) for item in x if str(item).strip()]
    return [str(x)] if str(x).strip() else []

def preprocess_data(df):
    # Drop the 'album' column as it's all NaN
    df = df.drop('album', axis=1)
    
    # Parse 'tags' and 'similar_tracks' columns
    df['tags'] = df['tags'].apply(robust_string_parser)
    df['similar_tracks'] = df['similar_tracks'].apply(robust_string_parser)
    
    # Create binary indicators for missing values
    df['has_tags'] = (df['tags'].apply(len) > 0).astype(int)
    df['has_similar_tracks'] = (df['similar_tracks'].apply(len) > 0).astype(int)
    
    return df

# Check for infinite values
def check_infinite_values(df):
    infinite_values = df.applymap(lambda x: np.isinf(x) if isinstance(x, (int, float)) else False).any().any()
    if infinite_values:
        print("Dataset contains infinite values.")
    else:
        print("No infinite values found in the dataset.")

# Check for very large values
def check_large_values(df, threshold=1e18):
    large_values = df.applymap(lambda x: abs(x) > threshold if isinstance(x, (int, float)) else False).any().any()
    if large_values:
        print(f"Dataset contains values larger than {threshold}.")
    else:
        print(f"No values larger than {threshold} found in the dataset.")


def one_hot_encode(series):
    unique_items = set(item for sublist in series for item in sublist)
    return pd.DataFrame([[1 if item in sublist else 0 for item in unique_items] for sublist in series],
                        columns=list(unique_items))


def impute_data(df, n_neighbors=10):
    # Prepare numeric data
    numeric_data = df[['playcount']].copy()
    
    # One-hot encode tags and similar_tracks
    tags_encoded = one_hot_encode(df['tags'])
    tracks_encoded = one_hot_encode(df['similar_tracks'])
    
    # Combine all features
    features = pd.concat([numeric_data, tags_encoded, tracks_encoded], axis=1)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Impute using KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_features = imputer.fit_transform(features_scaled)
    
    # Rescale imputed features
    imputed_features = scaler.inverse_transform(imputed_features)
    
    # Reconstruct dataframe
    imputed_df = pd.DataFrame(imputed_features, columns=features.columns, index=df.index)
    
    # Update original dataframe
    df['playcount'] = imputed_df['playcount']
    
    # Convert one-hot encoded back to lists
    tags_columns = tags_encoded.columns
    tracks_columns = tracks_encoded.columns
    
    df['tags'] = imputed_df[tags_columns].apply(lambda row: [col for col, val in row.items() if val > 0.5], axis=1)
    df['similar_tracks'] = imputed_df[tracks_columns].apply(lambda row: [col for col, val in row.items() if val > 0.5], axis=1)
    
    # Function to get most common tags/tracks for an artist
    def get_most_common(artist, column, n=3):
        artist_data = df[df['artist'] == artist][column]
        all_items = [item for sublist in artist_data for item in sublist if sublist]
        return Counter(all_items).most_common(n)
    
    # Get global most common tags and tracks
    global_common_tags = Counter([tag for tags in df['tags'] for tag in tags]).most_common(5)
    global_common_tracks = Counter([track for tracks in df['similar_tracks'] for track in tracks]).most_common(5)
    
    # Fill empty lists with most common tags/tracks for the artist or global common tags/tracks
    for idx, row in df.iterrows():
        if not row['tags']:
            common_tags = get_most_common(row['artist'], 'tags')
            if common_tags:
                df.at[idx, 'tags'] = [tag for tag, _ in common_tags]
            else:
                df.at[idx, 'tags'] = [tag for tag, _ in global_common_tags]
        
        if not row['similar_tracks']:
            common_tracks = get_most_common(row['artist'], 'similar_tracks')
            if common_tracks:
                df.at[idx, 'similar_tracks'] = [track for track, _ in common_tracks]
            else:
                df.at[idx, 'similar_tracks'] = [track for track, _ in global_common_tracks]
    
    # Convert lists back to strings
    df['tags'] = df['tags'].apply(lambda x: ', '.join(x) if x else 'Unknown')
    df['similar_tracks'] = df['similar_tracks'].apply(lambda x: ', '.join(x) if x else 'Unknown')
    
    # Update has_tags and has_similar_tracks
    df['has_tags'] = (df['tags'] != 'Unknown').astype(int)
    df['has_similar_tracks'] = (df['similar_tracks'] != 'Unknown').astype(int)
    
    return df

def get_value_counts(series):
    return pd.Series([item.strip() for items in series.dropna() for item in items.split(',') if item.strip()]).value_counts()

def analyze_list_column(df, column):
    all_items = [item.strip() for items in df[column] for item in items.split(',')]
    item_counts = pd.Series(all_items).value_counts()
    print(f"Top 10 most common {column}:")
    print(item_counts.head(10))
    print(f"\nNumber of unique {column}:", len(item_counts))

def analyze_playcount(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df['playcount'], kde=True)
    plt.title('Distribution of Playcount')
    plt.xlabel('Playcount')
    plt.ylabel('Frequency')
    plt.show()

    print("Playcount Statistics:")
    print(df['playcount'].describe())

    Q1 = df['playcount'].quantile(0.25)
    Q3 = df['playcount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['playcount'] < lower_bound) | (df['playcount'] > upper_bound)]
    print(f"\nNumber of outliers in playcount: {len(outliers)}")
    print("Sample of outliers:")
    print(outliers[['name', 'artist', 'playcount']].sample(min(5, len(outliers))))

def analyze_categorical_vs_playcount(df, top_n=20):
    # Artist vs Average Playcount
    artist_avg_playcount = df.groupby('artist')['playcount'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    artist_avg_playcount.head(top_n).plot(kind='bar')
    plt.title(f'Top {top_n} Artists by Average Playcount')
    plt.xlabel('Artist')
    plt.ylabel('Average Playcount')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Tags vs Average Playcount
    def get_top_tags(tags_series, top_n=20):
        all_tags = [tag.strip() for tags in tags_series for tag in tags.split(',')]
        return pd.Series(all_tags).value_counts().head(top_n).index

    top_tags = get_top_tags(df['tags'], top_n)
    tag_avg_playcount = {tag: df[df['tags'].str.contains(tag)]['playcount'].mean() for tag in top_tags}
    tag_avg_playcount = pd.Series(tag_avg_playcount).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    tag_avg_playcount.plot(kind='bar')
    plt.title(f'Average Playcount by Top {top_n} Tags')
    plt.xlabel('Tag')
    plt.ylabel('Average Playcount')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def analyze_co_occurrence(df, column, top_n=10):
    all_items = [set(items.split(', ')) for items in df[column]]
    co_occurrences = Counter()
    
    for items in all_items:
        co_occurrences.update(combinations(sorted(items), 2))
    
    co_occur = pd.Series(co_occurrences).sort_values(ascending=False).head(top_n)
    
    print(f"Top {top_n} co-occurring {column}:")
    print(co_occur)
    
    plt.figure(figsize=(12, 6))
    co_occur.plot(kind='bar')
    plt.title(f'Top {top_n} Co-occurring {column}')
    plt.xlabel(f'{column} Pairs')
    plt.ylabel('Co-occurrence Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def analyze_binary_vs_playcount(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='has_tags', y='playcount', data=df)
    plt.title('Playcount Distribution by has_tags')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='has_similar_tracks', y='playcount', data=df)
    plt.title('Playcount Distribution by has_similar_tracks')
    plt.show()

def analyze_tag_and_track_counts(df):
    analyzed_data = df.copy()
    analyzed_data.loc[:, 'num_tags'] = analyzed_data['tags'].apply(lambda x: len(x.split(', ')))
    analyzed_data.loc[:, 'num_similar_tracks'] = analyzed_data['similar_tracks'].apply(lambda x: len(x.split(', ')))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(analyzed_data['num_tags'], kde=True)
    plt.title('Distribution of Number of Tags')
    plt.subplot(1, 2, 2)
    sns.histplot(analyzed_data['num_similar_tracks'], kde=True)
    plt.title('Distribution of Number of Similar Tracks')
    plt.tight_layout()
    plt.show()

