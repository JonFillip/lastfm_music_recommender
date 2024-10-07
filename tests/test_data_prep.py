import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from google.cloud import bigquery
from src.data_processing.data_prep import (
    load_data_from_bigquery,
    prepare_data,
    save_prepared_data,
    create_and_populate_feature_store
)

class TestDataPrep(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            'artist': ['Artist1', 'Artist2'],
            'name': ['Song1', 'Song2'],
            'tags': ['rock, pop', 'jazz, blues'],
            'similar_tracks': ['Track1, Track2', 'Track3, Track4'],
            'playcount': [1000, 2000]
        })

    @patch('src.data_processing.data_prep.bigquery.Client')
    def test_load_data_from_bigquery(self, mock_client):
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = self.sample_df
        mock_client.return_value.query.return_value = mock_query_job

        result = load_data_from_bigquery('project_id', 'dataset_id', 'table_id')

        mock_client.assert_called_once_with(project='project_id')
        mock_client.return_value.query.assert_called_once()
        self.assertTrue(result.equals(self.sample_df))

    def test_prepare_data(self):
        preprocessed_df = self.sample_df.copy()
        preprocessed_df['extra_feature'] = [1, 2]
        
        X_train, X_test, y_train, y_test, names_train, names_test, scaler, mlb = prepare_data(preprocessed_df, self.sample_df)

        self.assertEqual(X_train.shape[1], 1)  # Only 'extra_feature' should be in X
        self.assertEqual(y_train.shape[1], 4)  # 4 unique tracks in similar_tracks
        self.assertEqual(len(names_train), 1)  # 80% of 2 samples
        self.assertEqual(len(names_test), 1)   # 20% of 2 samples

    @patch('src.data_processing.data_prep.np.save')
    @patch('src.data_processing.data_prep.joblib.dump')
    def test_save_prepared_data(self, mock_joblib_dump, mock_np_save):
        X_train = np.array([[1, 2], [3, 4]])
        X_test = np.array([[5, 6]])
        y_train = np.array([[1, 0], [0, 1]])
        y_test = np.array([[1, 0]])
        names_train = np.array(['Song1', 'Song2'])
        names_test = np.array(['Song3'])
        scaler = MagicMock()
        mlb = MagicMock()

        save_prepared_data(X_train, X_test, y_train, y_test, names_train, names_test, scaler, mlb, 'output_dir')

        self.assertEqual(mock_np_save.call_count, 6)  # 6 numpy arrays saved
        self.assertEqual(mock_joblib_dump.call_count, 2)  # scaler and mlb saved

    @patch('src.data_processing.data_prep.aiplatform.init')
    @patch('src.data_processing.data_prep.aiplatform.FeatureStore.create')
    def test_create_and_populate_feature_store(self, mock_create_feature_store, mock_init):
        mock_feature_store = MagicMock()
        mock_create_feature_store.return_value = mock_feature_store
        mock_entity_type = MagicMock()
        mock_feature_store.create_entity_type.return_value = mock_entity_type

        create_and_populate_feature_store('project_id', 'region', 'feature_store_id', 'entity_type_id', self.sample_df)

        mock_init.assert_called_once_with(project='project_id', location='region')
        mock_create_feature_store.assert_called_once()
        mock_feature_store.create_entity_type.assert_called_once()
        self.assertEqual(mock_entity_type.create_feature.call_count, 5)  # 5 features in sample_df
        mock_entity_type.ingest.assert_called_once()

if __name__ == '__main__':
    unittest.main()