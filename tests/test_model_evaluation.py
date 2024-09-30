import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import json
from src.evaluation.model_evaluation import (
    mean_average_precision,
    average_precision,
    diversity,
    novelty,
    evaluate_model,
    visualize_results,
    update_custom_metrics
)

class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        self.y_pred = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.6], [0.8, 0.3, 0.1]])
        self.item_popularity = {'item1': 0.5, 'item2': 0.3, 'item3': 0.2}

    def test_mean_average_precision(self):
        map_score = mean_average_precision(self.y_true, self.y_pred)
        self.assertIsInstance(map_score, float)
        self.assertTrue(0 <= map_score <= 1)

    def test_average_precision(self):
        ap_score = average_precision(self.y_true[0], self.y_pred[0])
        self.assertIsInstance(ap_score, float)
        self.assertTrue(0 <= ap_score <= 1)

    def test_diversity(self):
        recommendations = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        div_score = diversity(recommendations)
        self.assertIsInstance(div_score, float)
        self.assertTrue(0 <= div_score <= 1)

    def test_novelty(self):
        recommendations = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        nov_score = novelty(recommendations, self.item_popularity)
        self.assertIsInstance(nov_score, float)

    @patch('src.evaluation.model_evaluation.load_metrics')
    @patch('src.evaluation.model_evaluation.log_metric')
    def test_evaluate_model(self, mock_log_metric, mock_load_metrics):
        mock_model = MagicMock()
        mock_model.predict.return_value = self.y_pred
        mock_model.evaluate.return_value = (0.1, 0.9)  # mock loss and accuracy
        mock_load_metrics.return_value = {'train_loss': 0.2, 'train_accuracy': 0.8}

        results = evaluate_model(mock_model, self.y_true, self.y_true, 'mock_path', self.item_popularity)

        self.assertIsInstance(results, dict)
        self.assertIn('test_accuracy', results)
        self.assertIn('test_precision', results)
        self.assertIn('test_recall', results)
        self.assertIn('test_f1_score', results)
        self.assertIn('test_ndcg', results)
        self.assertIn('test_mean_average_precision', results)
        self.assertIn('test_diversity', results)
        self.assertIn('test_novelty', results)

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_results(self, mock_savefig):
        results = {'metric1': 0.5, 'metric2': 0.7}
        visualize_results(results, 'mock_output_path')
        mock_savefig.assert_called_once()

    @patch('src.evaluation.model_evaluation.monitoring_v3.MetricServiceClient')
    def test_update_custom_metrics(self, mock_client):
        metrics = {'accuracy': 0.9, 'f1_score': 0.8}
        update_custom_metrics('project_id', 'model_name', metrics)
        mock_client.return_value.create_time_series.assert_called()

if __name__ == '__main__':
    unittest.main()