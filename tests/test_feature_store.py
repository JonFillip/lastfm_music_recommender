import unittest
from unittest.mock import patch, MagicMock
from google.cloud import aiplatform
from src.data_processing.data_prep import create_and_populate_feature_store

class TestFeatureStore(unittest.TestCase):

    @patch('src.data_processing.data_prep.aiplatform.init')
    @patch('src.data_processing.data_prep.aiplatform.FeatureStore.create')
    def test_create_and_populate_feature_store(self, mock_create_feature_store, mock_init):
        mock_feature_store = MagicMock()
        mock_create_feature_store.return_value = mock_feature_store
        mock_entity_type = MagicMock()
        mock_feature_store.create_entity_type.return_value = mock_entity_type

        # Mock DataFrame
        df = MagicMock()
        df.to_dict.return_value = [{'feature1': 'value1', 'feature2': 'value2'}]
        df.index.tolist.return_value = ['entity1']

        create_and_populate_feature_store('project_id', 'region', 'feature_store_id', 'entity_type_id', df)

        mock_init.assert_called_once_with(project='project_id', location='region')
        mock_create_feature_store.assert_called_once()
        mock_feature_store.create_entity_type.assert_called_once()
        mock_entity_type.create_feature.assert_called()
        mock_entity_type.ingest.assert_called_once()

    # Add more tests for other feature store operations

if __name__ == '__main__':
    unittest.main()