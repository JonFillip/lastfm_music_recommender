import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.feature_engineering.feat_engineering import (
    engineer_basic_features,
    engineer_additional_features,
    add_tag_popularity,
    add_similar_tracks_avg_playcount,
    refine_features,
    vectorize_all_text_features
)

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            'artist': ['Artist1', 'Artist2'],
            'name': ['Song1', 'Song2'],
            'tags': ['rock, pop', 'jazz, blues'],
            'similar_tracks': ['Track1, Track2', 'Track3, Track4'],
            'playcount': [1000, 2000]
        })

    def test_engineer_basic_features(self):
        result = engineer_basic_features(self.sample_df)
        self.assertIn('log_playcount', result.columns)
        self.assertIn('num_tags', result.columns)
        self.assertIn('num_similar_tracks', result.columns)

    def test_engineer_additional_features(self):
        basic_features = engineer_basic_features(self.sample_df)
        result = engineer_additional_features(basic_features)
        self.assertIn('tag_count_category', result.columns)
        self.assertIn('similar_tracks_category', result.columns)

    @patch('src.feature_engineering.feat_engineering.pd.DataFrame.merge')
    def test_add_tag_popularity(self, mock_merge):
        mock_merge.return_value = self.sample_df
        result = add_tag_popularity(self.sample_df)
        self.assertIn('avg_tag_popularity', result.columns)

    def test_add_similar_tracks_avg_playcount(self):
        result = add_similar_tracks_avg_playcount(self.sample_df)
        self.assertIn('avg_similar_tracks_playcount', result.columns)

    def test_refine_features(self):
        result = refine_features(self.sample_df)
        self.assertIn('artist_avg_playcount', result.columns)
        self.assertIn('artist_track_count', result.columns)

    @patch('src.feature_engineering.feat_engineering.TfidfVectorizer')
    def test_vectorize_all_text_features(self, mock_tfidf):
        mock_tfidf.return_value.fit_transform.return_value.toarray.return_value = np.array([[1, 0], [0, 1]])
        result, _ = vectorize_all_text_features(self.sample_df)
        self.assertTrue(any(col.startswith('artist_tfidf_') for col in result.columns))
        self.assertTrue(any(col.startswith('name_tfidf_') for col in result.columns))
        self.assertTrue(any(col.startswith('tags_tfidf_') for col in result.columns))
        self.assertTrue(any(col.startswith('similar_tracks_tfidf_') for col in result.columns))

if __name__ == '__main__':
    unittest.main()