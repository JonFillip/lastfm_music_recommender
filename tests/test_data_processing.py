import unittest
import pandas as pd
import numpy as np
from src.data_processing.data_ingestion import fetch_lastfm_data
from src.data_processing.data_preprocess import (
    load_data, robust_string_parser, preprocess_data, one_hot_encode,
    impute_data, prepare_data, main
)
from src.data_processing.data_validation import validate_data
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import os
import tempfile

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

    def test_load_data(self):
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            self.sample_data.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name

        try:
            # Test loading the data
            loaded_data = load_data(temp_file_path)
            pd.testing.assert_frame_equal(loaded_data, self.sample_data)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    def test_robust_string_parser(self):
        test_cases = [
            ('rock, pop', ['rock', 'pop']),
            ('', []),
            (np.nan, []),
            (['rock', 'pop'], ['rock', 'pop']),
            (123, ['123']),
        ]
        for input_value, expected_output in test_cases:
            self.assertEqual(robust_string_parser(input_value), expected_output)

    def test_preprocess_data(self):
        preprocessed_data = preprocess_data(self.sample_data)
        
        # Check if 'album' column is dropped
        self.assertNotIn('album', preprocessed_data.columns)
        
        # Check if 'tags' and 'similar_tracks' are properly parsed
        self.assertTrue(all(isinstance(tags, list) for tags in preprocessed_data['tags']))
        self.assertTrue(all(isinstance(tracks, list) for tracks in preprocessed_data['similar_tracks']))
        
        # Check if binary indicators are created
        self.assertIn('has_tags', preprocessed_data.columns)
        self.assertIn('has_similar_tracks', preprocessed_data.columns)
        
        # Check if playcount is converted to numeric
        self.assertTrue(np.issubdtype(preprocessed_data['playcount'].dtype, np.number))

    def test_one_hot_encode(self):
        input_series = pd.Series([['A', 'B'], ['B', 'C'], ['A', 'C']])
        encoded = one_hot_encode(input_series)
        expected_output = pd.DataFrame({
            'A': [1, 0, 1],
            'B': [1, 1, 0],
            'C': [0, 1, 1]
        })
        pd.testing.assert_frame_equal(encoded, expected_output)

    def test_impute_data(self):
        # Create a sample DataFrame with missing values
        df_with_missing = pd.DataFrame({
            'name': ['Song1', 'Song2', 'Song3'],
            'artist': ['Artist1', 'Artist2', 'Artist3'],
            'playcount': [1000, np.nan, 3000],
            'tags': [['rock', 'pop'], [], ['electronic', 'dance']],
            'similar_tracks': [['Track1', 'Track2'], ['Track3', 'Track4'], []]
        })
        
        imputed_df = impute_data(df_with_missing)
        
        # Check if missing values are imputed
        self.assertFalse(imputed_df['playcount'].isnull().any())
        self.assertTrue(all(len(tags) > 0 for tags in imputed_df['tags']))
        self.assertTrue(all(len(tracks) > 0 for tracks in imputed_df['similar_tracks']))

    def test_prepare_data(self):
        preprocessed_df = preprocess_data(self.sample_data)
        X_train, X_test, y_train, y_test, names_train, names_test, scaler, mlb = prepare_data(preprocessed_df, self.sample_data)
        
        # Check shapes
        self.assertEqual(X_train.shape[0] + X_test.shape[0], len(self.sample_data))
        self.assertEqual(y_train.shape[0] + y_test.shape[0], len(self.sample_data))
        
        # Check if data is scaled
        self.assertTrue(np.all(X_train.mean(axis=0) < 1e-6))  # Close to 0 mean
        self.assertTrue(np.all(np.abs(X_train.std(axis=0) - 1) < 1e-6))  # Close to unit variance
        
        # Check if MultiLabelBinarizer is properly fitted
        self.assertTrue(isinstance(mlb, MultiLabelBinarizer))
        self.assertTrue(len(mlb.classes_) > 0)

    def test_main(self):
        # Create temporary input and output files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file, \
            tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
            self.sample_data.to_csv(input_file.name, index=False)
            input_path = input_file.name
            output_path = output_file.name

        try:
            # Run the main function
            main(input_path, output_path)
            
            # Check if the output file is created and not empty
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(os.path.getsize(output_path) > 0)
            
            # Load the output file and perform some basic checks
            output_data = pd.read_csv(output_path)
            self.assertEqual(len(output_data), len(self.sample_data))
            self.assertNotIn('album', output_data.columns)
            self.assertIn('has_tags', output_data.columns)
            self.assertIn('has_similar_tracks', output_data.columns)
        finally:
            # Clean up temporary files
            os.unlink(input_path)
            os.unlink(output_path)

if __name__ == '__main__':
    unittest.main()