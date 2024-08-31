import unittest
import pandas as pd
import numpy as np
from src.feature_engineering.feat_engineering import (engineer_basic_features, engineer_additional_features,
                        refine_features, add_tag_popularity,
                        add_similar_tracks_avg_playcount, add_interaction_features,
                        add_target_encoding, refine_features_further,
                        vectorize_all_text_features)

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'name': ['Track1', 'Track2', 'Track3'],
            'artist': ['Artist1', 'Artist2', 'Artist1'],
            'playcount': [100, 200, 300],
            'tags': ['rock, pop', 'jazz, blues', 'rock, metal'],
            'similar_tracks': ['Track2, Track3', 'Track1, Track3', 'Track1, Track2']
        })

    def test_engineer_basic_features(self):
        result = engineer_basic_features(self.df)
        
        # Check exact values
        np.testing.assert_almost_equal(result['log_playcount'].values, 
                                    np.log1p([100, 200, 300]))
        
        # Check types
        self.assertTrue(np.issubdtype(result['log_playcount'].dtype, np.number))
        self.assertTrue(np.issubdtype(result['binned_playcount'].dtype, np.integer))
        
        # Check edge case: empty DataFrame
        empty_df = pd.DataFrame(columns=self.df.columns)
        empty_result = engineer_basic_features(empty_df)
        self.assertTrue(empty_result.empty)

    def test_engineer_additional_features(self):
        df_basic = engineer_basic_features(self.df)
        result = engineer_additional_features(df_basic)
        
        # Check exact values
        self.assertEqual(result['high_tag_count'].tolist(), [0, 0, 0])
        self.assertEqual(result['tag_count_category'].tolist(), ['low', 'low', 'low'])
        
        # Check types
        self.assertTrue(np.issubdtype(result['high_tag_count'].dtype, np.integer))
        self.assertTrue(pd.api.types.is_categorical_dtype(result['tag_count_category']))

    def test_refine_features(self):
        df_basic = engineer_basic_features(self.df)
        df_additional = engineer_additional_features(df_basic)
        result = refine_features(df_additional)
        
        # Check exact values
        np.testing.assert_almost_equal(result['artist_avg_playcount'].values, 
                                    [np.log1p(200), np.log1p(200), np.log1p(200)])
        
        # Check types
        self.assertTrue(np.issubdtype(result['artist_track_count'].dtype, np.integer))

    def test_add_tag_popularity(self):
        result = add_tag_popularity(self.df)
        
        # Check if tag popularity is calculated correctly
        self.assertGreater(result.loc[0, 'avg_tag_popularity'], 
                            result.loc[1, 'avg_tag_popularity'])
        
        # Test with missing values
        df_with_missing = self.df.copy()
        df_with_missing.loc[0, 'tags'] = np.nan
        result_missing = add_tag_popularity(df_with_missing)
        self.assertEqual(result_missing.loc[0, 'avg_tag_popularity'], 0)

    def test_add_similar_tracks_avg_playcount(self):
        result = add_similar_tracks_avg_playcount(self.df)
        
        # Check exact values
        expected_avg = (np.log1p(200) + np.log1p(300)) / 2
        self.assertAlmostEqual(result.loc[0, 'avg_similar_tracks_playcount'], expected_avg)

    def test_add_interaction_features(self):
        df_with_avg = add_similar_tracks_avg_playcount(self.df)
        df_with_avg['num_tags'] = df_with_avg['tags'].str.count(',') + 1
        result = add_interaction_features(df_with_avg)
        
        # Check exact values
        expected_interaction = (df_with_avg['num_tags'] * df_with_avg['avg_similar_tracks_playcount']).values
        np.testing.assert_almost_equal(result['num_tags_x_avg_similar_tracks_playcount'].values, expected_interaction)

    def test_add_target_encoding(self):
        df_with_log = engineer_basic_features(self.df)
        result = add_target_encoding(df_with_log)
        
        # Check if encoding is smooth
        self.assertNotEqual(result['artist_target_encoded'].nunique(), result['artist'].nunique())

    def test_refine_features_further(self):
        df_refined = refine_features(self.df)
        df_refined['has_tag_favorites'] = [1, 0, 1]
        df_refined['has_tag_Favorite'] = [0, 1, 0]
        df_refined['has_tag_MySpotigramBot'] = [1, 1, 1]
        result = refine_features_further(df_refined)
        
        # Check exact values
        np.testing.assert_array_equal(result['has_tag_favorites_combined'].values, [1, 1, 1])
        
        # Check if low variance feature is dropped
        self.assertNotIn('has_tag_MySpotigramBot', result.columns)

    def test_vectorize_all_text_features(self):
        result, vectorizers = vectorize_all_text_features(self.df)
        
        # Check if vectorization produces expected number of features
        self.assertEqual(sum(1 for col in result.columns if col.startswith('name_tfidf_')), 3)
        self.assertEqual(sum(1 for col in result.columns if col.startswith('artist_tfidf_')), 2)
        
        # Test with custom max_features
        result_custom, _ = vectorize_all_text_features(self.df, {'tags': 1, 'similar_tracks': 1})
        self.assertEqual(sum(1 for col in result_custom.columns if col.startswith('tags_tfidf_')), 1)
        self.assertEqual(sum(1 for col in result_custom.columns if col.startswith('similar_tracks_tfidf_')), 1)

    def test_edge_cases(self):
        # Test with empty DataFrame
        empty_df = pd.DataFrame(columns=self.df.columns)
        self.assertTrue(engineer_basic_features(empty_df).empty)
        self.assertTrue(engineer_additional_features(empty_df).empty)
        self.assertTrue(refine_features(empty_df).empty)
        
        # Test with missing values
        df_with_missing = self.df.copy()
        df_with_missing.loc[0, 'playcount'] = np.nan
        df_with_missing.loc[1, 'tags'] = np.nan
        df_with_missing.loc[2, 'similar_tracks'] = np.nan
        
        result_basic = engineer_basic_features(df_with_missing)
        self.assertTrue(np.isnan(result_basic.loc[0, 'log_playcount']))
        self.assertEqual(result_basic.loc[1, 'num_tags'], 0)
        self.assertEqual(result_basic.loc[2, 'num_similar_tracks'], 0)

if __name__ == '__main__':
    unittest.main()