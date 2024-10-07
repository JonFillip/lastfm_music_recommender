import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tensorflow as tf
from src.algorithms.content_based import (
    cosine_similarity,
    mean_average_precision,
    average_precision,
    FilteredCallback,
    train_model,
    evaluate_model,
    find_similar_tracks,
    main
)

class TestContentBased(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        self.y_pred = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.6], [0.8, 0.3, 0.1]])

    def test_cosine_similarity(self):
        similarity = cosine_similarity(self.y_true, self.y_pred)
        self.assertIsInstance(similarity, tf.Tensor)

    def test_mean_average_precision(self):
        map_score = mean_average_precision(self.y_true, self.y_pred)
        self.assertIsInstance(map_score, float)
        self.assertTrue(0 <= map_score <= 1)

    def test_average_precision(self):
        ap_score = average_precision(self.y_true[0], self.y_pred[0])
        self.assertIsInstance(ap_score, float)
        self.assertTrue(0 <= ap_score <= 1)

    @patch('src.algorithms.content_based.tf.keras.callbacks.ModelCheckpoint')
    def test_filtered_callback(self, mock_model_checkpoint):
        filtered_callback = FilteredCallback(filepath='test_path')
        self.assertIsInstance(filtered_callback, tf.keras.callbacks.ModelCheckpoint)

    @patch('src.algorithms.content_based.EarlyStopping')
    @patch('src.algorithms.content_based.FilteredCallback')
    def test_train_model(self, mock_filtered_callback, mock_early_stopping):
        mock_model = MagicMock()
        mock_model.fit.return_value = MagicMock(history={'loss': [0.1], 'val_loss': [0.2]})
        
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100, 3)
        X_val = np.random.rand(20, 10)
        y_val = np.random.rand(20, 3)

        history = train_model(mock_model, X_train, y_train, X_val, y_val)
        
        mock_model.fit.assert_called_once()
        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)

    @patch('src.algorithms.content_based.precision_score')
    @patch('src.algorithms.content_based.recall_score')
    @patch('src.algorithms.content_based.f1_score')
    @patch('src.algorithms.content_based.ndcg_score')
    def test_evaluate_model(self, mock_ndcg, mock_f1, mock_recall, mock_precision):
        mock_model = MagicMock()
        mock_model.predict.return_value = self.y_pred
        mock_model.evaluate.return_value = [0.1, 0.9]
        
        mock_precision.return_value = 0.8
        mock_recall.return_value = 0.7
        mock_f1.return_value = 0.75
        mock_ndcg.return_value = 0.85

        metrics = evaluate_model(mock_model, self.y_true, self.y_true)
        
        self.assertIn('test_loss', metrics)
        self.assertIn('test_accuracy', metrics)
        self.assertIn('cosine_similarity', metrics)
        self.assertIn('mean_average_precision', metrics)
        self.assertIn('ndcg', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)

    def test_find_similar_tracks(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = [
            np.array([[0.1, 0.2, 0.3]]),  # track_embedding
            np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])  # all_embeddings
        ]
        
        track_features = np.array([1, 2, 3])
        all_features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        track_names = ['Track1', 'Track2', 'Track3']

        similar_tracks = find_similar_tracks(mock_model, track_features, all_features, track_names, n=2)
        
        self.assertEqual(len(similar_tracks), 2)
        self.assertIsInstance(similar_tracks[0], tuple)
        self.assertIsInstance(similar_tracks[0][0], str)
        self.assertIsInstance(similar_tracks[0][1], float)

    @patch('src.algorithms.content_based.prepare_data')
    @patch('src.algorithms.content_based.build_content_based_model')
    @patch('src.algorithms.content_based.train_model')
    @patch('src.algorithms.content_based.evaluate_model')
    @patch('src.algorithms.content_based.json.dump')
    def test_main(self, mock_json_dump, mock_evaluate, mock_train, mock_build, mock_prepare):
        mock_prepare.return_value = (
            np.random.rand(100, 10), np.random.rand(20, 10), np.random.rand(30, 10),
            np.random.rand(100, 3), np.random.rand(20, 3), np.random.rand(30, 3),
            ['name1', 'name2'], ['name3'], ['name4', 'name5'],
            MagicMock(), MagicMock()
        )
        
        mock_model = MagicMock()
        mock_build.return_value = mock_model
        
        mock_history = MagicMock()
        mock_history.history = {
            'loss': [0.1], 'binary_accuracy': [0.9],
            'val_loss': [0.2], 'val_binary_accuracy': [0.8],
            'val_cosine_similarity': [-0.7]
        }
        mock_train.return_value = mock_history
        
        mock_evaluate.return_value = {
            'test_loss': 0.15, 'test_accuracy': 0.85,
            'cosine_similarity': 0.8, 'mean_average_precision': 0.75,
            'ndcg': 0.9, 'precision': 0.8, 'recall': 0.7, 'f1_score': 0.75
        }

        model, metrics = main('feat_eng_data.csv', 'original_df.csv', 2, 64, 32, 0.001, 32, 0.2)
        
        mock_prepare.assert_called_once()
        mock_build.assert_called_once()
        mock_train.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_json_dump.assert_called_once()
        
        self.assertIsInstance(model, MagicMock)
        self.assertIsInstance(metrics, dict)
        self.assertIn('final_loss', metrics)
        self.assertIn('final_accuracy', metrics)
        self.assertIn('final_val_loss', metrics)
        self.assertIn('final_val_accuracy', metrics)
        self.assertIn('val_cosine_similarity', metrics)
        self.assertIn('test_loss', metrics)
        self.assertIn('test_accuracy', metrics)
        self.assertIn('cosine_similarity', metrics)
        self.assertIn('mean_average_precision', metrics)
        self.assertIn('ndcg', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)

if __name__ == '__main__':
    unittest.main()