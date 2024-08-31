import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from io import StringIO
import requests

# Import the functions you want to test
from src.data_processing.data_ingestion import (configure_lastfm_api, fetch_track_details, 
                        fetch_lastfm_data, main)

class TestLastFMDataIngestion(unittest.TestCase):

    def setUp(self):
        # Set up any necessary test data or configurations
        self.api_key = "test_api_key"
        self.api_secret = "test_api_secret"

    @patch.dict(os.environ, {"LASTFM_API_KEY": "test_api_key", "LASTFM_API_SECRET": "test_api_secret"})
    def test_configure_lastfm_api(self):
        api_key, api_secret = configure_lastfm_api()
        self.assertEqual(api_key, "test_api_key")
        self.assertEqual(api_secret, "test_api_secret")

    @patch.dict(os.environ, {})
    def test_configure_lastfm_api_missing_env_vars(self):
        with self.assertRaises(ValueError):
            configure_lastfm_api()

    @patch('requests.get')
    def test_fetch_track_details(self, mock_get):
        # Mock the API responses
        mock_tags_response = MagicMock()
        mock_tags_response.json.return_value = {
            "toptags": {"tag": [{"name": "rock"}, {"name": "pop"}]}
        }
        mock_similar_response = MagicMock()
        mock_similar_response.json.return_value = {
            "similartracks": {"track": [{"name": "Similar Track 1"}, {"name": "Similar Track 2"}]}
        }
        mock_get.side_effect = [mock_tags_response, mock_similar_response]

        tags, similar_tracks = fetch_track_details(self.api_key, "Test Track", "Test Artist")
        
        self.assertEqual(tags, ["rock", "pop"])
        self.assertEqual(similar_tracks, ["Similar Track 1", "Similar Track 2"])

    @patch('requests.get')
    def test_fetch_track_details_http_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("HTTP Error")
        tags, similar_tracks = fetch_track_details(self.api_key, "Test Track", "Test Artist")
        self.assertEqual(tags, [])
        self.assertEqual(similar_tracks, [])

    @patch('requests.get')
    def test_fetch_lastfm_data(self, mock_get):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tracks": {
                "track": [
                    {
                        "name": "Track 1",
                        "artist": {"name": "Artist 1"},
                        "album": {"title": "Album 1"},
                        "playcount": "1000"
                    },
                    {
                        "name": "Track 2",
                        "artist": {"name": "Artist 2"},
                        "playcount": "2000"
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        with patch('src.data_processing.data_ingestion.fetch_track_details', return_value=(["rock"], ["Similar Track"])):
            df = fetch_lastfm_data(self.api_key, limit=2)

        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ['name', 'artist', 'album', 'playcount', 'tags', 'similar_tracks'])
        self.assertEqual(df.iloc[0]['name'], "Track 1")
        self.assertEqual(df.iloc[1]['album'], None)  # Test handling of missing album

    @patch('requests.get')
    def test_fetch_lastfm_data_error(self, mock_get):
        mock_get.side_effect = Exception("API Error")
        df = fetch_lastfm_data(self.api_key)
        self.assertTrue(df.empty)

    @patch('src.data_processing.data_ingestion.configure_lastfm_api')
    @patch('src.data_processing.data_ingestion.fetch_lastfm_data')
    def test_main(self, mock_fetch_data, mock_configure_api):
        mock_configure_api.return_value = (self.api_key, self.api_secret)
        mock_df = pd.DataFrame({
            'name': ['Track 1', 'Track 2'],
            'artist': ['Artist 1', 'Artist 2'],
            'album': ['Album 1', 'Album 2'],
            'playcount': [1000, 2000],
            'tags': ['rock, pop', 'jazz, blues'],
            'similar_tracks': ['Similar 1, Similar 2', 'Similar 3, Similar 4']
        })
        mock_fetch_data.return_value = mock_df

        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            main('test_output.csv')
            mock_to_csv.assert_called_once()

    @patch('src.data_processing.data_ingestion.configure_lastfm_api')
    @patch('src.data_processing.data_ingestion.fetch_lastfm_data')
    def test_main_empty_dataframe(self, mock_fetch_data, mock_configure_api):
        mock_configure_api.return_value = (self.api_key, self.api_secret)
        mock_fetch_data.return_value = pd.DataFrame()

        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            main('test_output.csv')
            mock_to_csv.assert_not_called()

if __name__ == '__main__':
    unittest.main()