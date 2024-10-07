import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.data_processing.data_ingestion import load_data
from src.data_processing.data_process import preprocess_data
from src.feature_engineering.feat_engineering import engineer_features
from src.algorithms.content_based import main as content_based_main
from src.evaluation.model_evaluation import evaluate_model

class TestIntegration(unittest.TestCase):

    @patch('src.data_processing.data_ingestion.load_data')
    @patch('src.data_processing.data_process.preprocess_data')
    @patch('src.feature_engineering.feat_engineering.engineer_features')
    @patch('src.algorithms.content_based.main')
    @patch('src.evaluation.model_evaluation.evaluate_model')
    def test_end_to_end_workflow(self, mock_evaluate, mock_content_based, mock_engineer, mock_preprocess, mock_load):
        # Mock data ingestion
        mock_load.return_value = pd.DataFrame({
            'user_id': [1, 2, 3],
            'track_id': [101, 102, 103],
            'listen_count': [10, 20, 30]
        })

        # Mock data preprocessing
        mock_preprocess.return_value = pd.DataFrame({
            'user_id': [1, 2, 3],
            'track_id': [101, 102, 103],
            'listen_count': [10, 20, 30],
            'normalized_listen_count': [0.1, 0.2, 0.3]
        })

        # Mock feature engineering
        mock_engineer.return_value = pd.DataFrame({
            'user_id': [1, 2, 3],
            'track_id': [101, 102, 103],
            'listen_count': [10, 20, 30],
            'normalized_listen_count': [0.1, 0.2, 0.3],
            'feature1': [0.5, 0.6, 0.7],
            'feature2': [0.8, 0.9, 1.0]
        })

        # Mock content-based algorithm
        mock_model = MagicMock()
        mock_content_based.return_value = (mock_model, {'accuracy': 0.85, 'f1_score': 0.82})

        # Mock model evaluation
        mock_evaluate.return_value = {'accuracy': 0.87, 'f1_score': 0.84, 'precision': 0.86, 'recall': 0.85}

        # Run the end-to-end workflow
        raw_data = load_data('dummy_path')
        preprocessed_data = preprocess_data(raw_data)
        feature_engineered_data = engineer_features(preprocessed_data)
        model, training_metrics = content_based_main(feature_engineered_data, preprocessed_data, 2, 64, 32, 0.001, 32, 0.2)
        evaluation_metrics = evaluate_model(model, feature_engineered_data, preprocessed_data)

        # Assertions
        self.assertIsNotNone(raw_data)
        self.assertIsNotNone(preprocessed_data)
        self.assertIsNotNone(feature_engineered_data)
        self.assertIsNotNone(model)
        self.assertIsNotNone(training_metrics)
        self.assertIsNotNone(evaluation_metrics)

        self.assertIn('accuracy', training_metrics)
        self.assertIn('f1_score', training_metrics)
        self.assertIn('accuracy', evaluation_metrics)
        self.assertIn('f1_score', evaluation_metrics)
        self.assertIn('precision', evaluation_metrics)
        self.assertIn('recall', evaluation_metrics)

        # Verify that each step was called with the output of the previous step
        mock_preprocess.assert_called_once_with(raw_data)
        mock_engineer.assert_called_once_with(mock_preprocess.return_value)
        mock_content_based.assert_called_once_with(mock_engineer.return_value, mock_preprocess.return_value, 2, 64, 32, 0.001, 32, 0.2)
        mock_evaluate.assert_called_once_with(mock_model, mock_engineer.return_value, mock_preprocess.return_value)

if __name__ == '__main__':
    unittest.main()