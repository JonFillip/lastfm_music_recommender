import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import tensorflow_data_validation as tfdv
from src.data_processing.data_validation import (
    generate_schema,
    validate_data,
    compare_statistics,
    detect_data_drift,
    compare_schemas
)

class TestDataValidation(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            'artist': ['Artist1', 'Artist2'],
            'name': ['Song1', 'Song2'],
            'tags': ['rock, pop', 'jazz, blues'],
            'similar_tracks': ['Track1, Track2', 'Track3, Track4'],
            'playcount': [1000, 2000]
        })

    @patch('src.data_processing.data_validation.tfdv.infer_schema')
    @patch('src.data_processing.data_validation.save_schema_to_gcs')
    def test_generate_schema(self, mock_save_schema, mock_infer_schema):
        mock_schema = MagicMock()
        mock_infer_schema.return_value = mock_schema
        
        result = generate_schema('project_id', 'dataset_id', 'table_id', 'bucket_name', 'model_name', 'version')
        
        mock_infer_schema.assert_called_once()
        mock_save_schema.assert_called_once_with(mock_schema, 'bucket_name', 'model_name', 'version')
        self.assertEqual(result, mock_schema)

    @patch('src.data_processing.data_validation.tfdv.generate_statistics_from_dataframe')
    @patch('src.data_processing.data_validation.tfdv.validate_statistics')
    @patch('src.data_processing.data_validation.save_statistics_to_gcs')
    def test_validate_data(self, mock_save_stats, mock_validate_stats, mock_generate_stats):
        mock_stats = MagicMock()
        mock_generate_stats.return_value = mock_stats
        mock_anomalies = MagicMock()
        mock_validate_stats.return_value = mock_anomalies
        mock_schema = MagicMock()

        stats, anomalies = validate_data('project_id', 'dataset_id', 'table_id', mock_schema, 'bucket_name', 'model_name', 'data_type')

        mock_generate_stats.assert_called_once()
        mock_save_stats.assert_called_once_with(mock_stats, 'bucket_name', 'model_name', 'data_type')
        mock_validate_stats.assert_called_once_with(mock_stats, mock_schema)
        self.assertEqual(stats, mock_stats)
        self.assertEqual(anomalies, mock_anomalies)

    @patch('src.data_processing.data_validation.tfdv.validate_statistics')
    def test_compare_statistics(self, mock_validate_stats):
        mock_train_stats = MagicMock()
        mock_serving_stats = MagicMock()
        mock_schema = MagicMock()
        mock_anomalies = MagicMock()
        mock_validate_stats.return_value = mock_anomalies

        result = compare_statistics(mock_train_stats, mock_serving_stats, mock_schema)

        mock_validate_stats.assert_called_once_with(mock_serving_stats, mock_schema, previous_statistics=mock_train_stats)
        self.assertEqual(result, mock_anomalies)

    @patch('src.data_processing.data_validation.tfdv.compute_drift_skew')
    def test_detect_data_drift(self, mock_compute_drift_skew):
        mock_train_stats = MagicMock()
        mock_serving_stats = MagicMock()
        mock_schema = MagicMock()
        mock_drift_skew = {'feature1': 0.1, 'feature2': 0.2}
        mock_compute_drift_skew.return_value = mock_drift_skew

        result = detect_data_drift(mock_train_stats, mock_serving_stats, mock_schema, 0.15)

        mock_compute_drift_skew.assert_called_once_with(mock_train_stats, mock_serving_stats, mock_schema)
        self.assertEqual(result, mock_drift_skew)

    def test_compare_schemas(self):
        baseline_schema = tfdv.Schema()
        baseline_schema.feature.add(name='feature1', type=tfdv.FeatureType.INT)
        baseline_schema.feature.add(name='feature2', type=tfdv.FeatureType.FLOAT)

        current_schema = tfdv.Schema()
        current_schema.feature.add(name='feature1', type=tfdv.FeatureType.INT)
        current_schema.feature.add(name='feature3', type=tfdv.FeatureType.STRING)

        result = compare_schemas(baseline_schema, current_schema)

        self.assertTrue(result)  # Schema drift detected

if __name__ == '__main__':
    unittest.main()