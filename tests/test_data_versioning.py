import unittest
from unittest.mock import patch, MagicMock
from src.utils.data_versioning import version_dataset

class TestDataVersioning(unittest.TestCase):
    @patch('src.utils.data_versioning.bigquery.Client')
    def test_version_dataset(self, mock_client):
        # Mock the BigQuery client
        mock_job = MagicMock()
        mock_client.return_value.query.return_value = mock_job

        # Call the function
        result = version_dataset('test-project', 'source_dataset', 'source_table', 'versioned_dataset')

        # Assert that the query was called with the correct parameters
        mock_client.return_value.query.assert_called_once()
        query_call = mock_client.return_value.query.call_args[0][0]
        self.assertIn('test-project.versioned_dataset', query_call)
        self.assertIn('test-project.source_dataset.source_table', query_call)

        # Assert that the job's result method was called
        mock_job.result.assert_called_once()

        # Assert that the function returns the correct table name
        self.assertTrue(result.startswith('versioned_dataset.source_table_v'))

if __name__ == '__main__':
    unittest.main()