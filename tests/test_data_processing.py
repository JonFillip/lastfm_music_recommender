import unittest
import pandas as pd
import numpy as np
from src.data_processing.data_ingestion import fetch_lastfm_data
from src.data_processing.data_preprocess import preprocess_data
from src.data_processing.data_validation import validate_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'name': ['Song1', 'Song2', 'Song3'],
            'artist': ['Artist1', 'Artist2', 'Artist3'],
            'album': ['Album1', 'Album2', 'Album3'],
            'playcount': ['1000', '2000', '3000'],
            'tags': ['rock, pop', 'jazz, blues', 'electronic, dance'],
            'similar_tracks': ['Track1, Track2', 'Track3, Track4', 'Track5, Track6']
        })

    def test_data_ingestion(self):
        # Mock the API call for testing
        def mock_fetch_lastfm_data(api_key, limit):
            return self.sample_data

        # Test the data ingestion function
        result = mock_fetch_lastfm_data('fake_api_key', 3)
        self.assertEqual(len(result), 3)
        self.assertListEqual(list(result.columns), ['name', 'artist', 'album', 'playcount', 'tags', 'similar_tracks'])

    def test_data_preprocessing(self):
        # Test the data preprocessing function
        processed_data, mlb = preprocess_data(self.sample_data)
        
        # Check if playcount is normalized
        self.assertTrue('playcount_normalized' in processed_data.columns)
        self.assertAlmostEqual(processed_data['playcount_normalized'].mean(), 0, places=7)
        self.assertAlmostEqual(processed_data['playcount_normalized'].std(), 1, places=7)

        # Check if tags are one-hot encoded
        self.assertTrue('tag_rock' in processed_data.columns)
        self.assertTrue('tag_pop' in processed_data.columns)
        self.assertTrue('tag_jazz' in processed_data.columns)

    def test_data_validation(self):
        # Create a schema for testing
        schema = {
            'name': {'type': 'string'},
            'artist': {'type': 'string'},
            'album': {'type': 'string'},
            'playcount': {'type': 'integer'},
            'tags': {'type': 'string'},
            'similar_tracks': {'type': 'string'}
        }

        # Test the data validation function
        anomalies = validate_data(self.sample_data, schema)
        self.assertFalse(anomalies.anomaly_info)  # Expecting no anomalies in the sample data

if __name__ == '__main__':
    unittest.main()