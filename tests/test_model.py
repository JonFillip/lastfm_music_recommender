import unittest
import numpy as np
import tensorflow as tf
from src.algorithms.content_based import ContentBasedRecommender
from src.evaluation.model_evaluation import evaluate_model

class TestModel(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.X_train = np.random.rand(100, 10)  # 100 samples, 10 features
        self.y_train = np.random.rand(100, 5)   # 100 samples, 5 target values
        self.X_test = np.random.rand(20, 10)    # 20 samples, 10 features
        self.y_test = np.random.rand(20, 5)     # 20 samples, 5 target values

    def test_model_creation(self):
        model = ContentBasedRecommender(input_dim=10, output_dim=5)
        self.assertIsInstance(model.model, tf.keras.Model)
        self.assertEqual(model.model.input_shape[1:], (10,))
        self.assertEqual(model.model.output_shape[1:], (5,))

    def test_model_training(self):
        model = ContentBasedRecommender(input_dim=10, output_dim=5)
        history = model.fit(self.X_train, self.y_train, epochs=5, validation_split=0.2)
        
        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)
        self.assertEqual(len(history.history['loss']), 5)

    def test_model_prediction(self):
        model = ContentBasedRecommender(input_dim=10, output_dim=5)
        model.fit(self.X_train, self.y_train, epochs=5)
        
        predictions = model.predict(self.X_test)
        self.assertEqual(predictions.shape, (20, 5))

    def test_model_evaluation(self):
        model = ContentBasedRecommender(input_dim=10, output_dim=5)
        model.fit(self.X_train, self.y_train, epochs=5)
        
        config = {
            'model_evaluation': {
                'metrics': ['mse', 'mae']
            }
        }
        
        results = evaluate_model(model.model, self.X_test, self.y_test, config)
        
        self.assertIn('mse', results)
        self.assertIn('mae', results)
        self.assertTrue(isinstance(results['mse'], float))
        self.assertTrue(isinstance(results['mae'], float))

if __name__ == '__main__':
    unittest.main()