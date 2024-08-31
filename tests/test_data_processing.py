import unittest
import pandas as pd
import numpy as np
from src.data_processing.data_ingestion import fetch_lastfm_data
from src.data_processing.data_preprocess import preprocess_data, split_data
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
        
        # Check if all expected columns are present
        expected_columns = ['name', 'artist', 'playcount_normalized'] + [f'tag_{tag}' for tag in mlb.classes_]
        self.assertTrue(all(col in processed_data.columns for col in expected_columns))
        
        # Check if playcount is normalized
        self.assertTrue(processed_data['playcount_normalized'].min() >= 0)
        self.assertTrue(processed_data['playcount_normalized'].max() <= 1)
        
        # Check if tags are one-hot encoded
        self.assertTrue(all(processed_data[f'tag_{tag}'].isin([0, 1]).all() for tag in mlb.classes_))
        
        # Check if 'album' column is dropped
        self.assertNotIn('album', processed_data.columns)
        
        # Check if binary indicators for missing values are created
        self.assertIn('has_tags', processed_data.columns)
        self.assertIn('has_similar_tracks', processed_data.columns)

    def test_data_split(self):
        processed_data, _ = preprocess_data(self.sample_data)
        train, val, test = split_data(processed_data)
        
        # Check if the splits have the correct proportions
        self.assertAlmostEqual(len(train) / len(processed_data), 0.6, delta=0.1)
        self.assertAlmostEqual(len(val) / len(processed_data), 0.2, delta=0.1)
        self.assertAlmostEqual(len(test) / len(processed_data), 0.2, delta=0.1)
        
        # Check if the splits have the same columns as the input data
        self.assertListEqual(list(train.columns), list(processed_data.columns))
        self.assertListEqual(list(val.columns), list(processed_data.columns))
        self.assertListEqual(list(test.columns), list(processed_data.columns))

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
        
        # Test with invalid data
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'playcount'] = 'invalid'
        anomalies = validate_data(invalid_data, schema)
        self.assertTrue(anomalies.anomaly_info)  # Expecting anomalies in the invalid data

if __name__ == '__main__':
    unittest.main()