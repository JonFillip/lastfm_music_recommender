import unittest
import numpy as np
import tensorflow as tf
from src.algorithms.content_based import ContentBasedRecommender
from src.evaluation.model_evaluation import evaluate_model
import tempfile
import os
import time

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

    def test_model_output_types_and_shapes(self):
        model = ContentBasedRecommender(input_dim=10, output_dim=5)
        
        # Test single input
        single_input = np.random.rand(1, 10)
        single_output = model.predict(single_input)
        self.assertEqual(single_output.shape, (1, 5))
        self.assertTrue(np.issubdtype(single_output.dtype, np.floating))
        
        # Test batch input
        batch_input = np.random.rand(32, 10)
        batch_output = model.predict(batch_input)
        self.assertEqual(batch_output.shape, (32, 5))
        self.assertTrue(np.issubdtype(batch_output.dtype, np.floating))

    def test_model_fit_on_small_batches(self):
        model = ContentBasedRecommender(input_dim=10, output_dim=5)
        
        # Test on two small batches
        batch1 = (np.random.rand(5, 10), np.random.rand(5, 5))
        batch2 = (np.random.rand(5, 10), np.random.rand(5, 5))
        
        # First batch
        history1 = model.fit(batch1[0], batch1[1], epochs=5, verbose=0)
        loss1 = history1.history['loss']
        
        # Second batch
        history2 = model.fit(batch2[0], batch2[1], epochs=5, verbose=0)
        loss2 = history2.history['loss']
        
        # Check if loss decreases
        self.assertTrue(loss1[0] > loss1[-1])
        self.assertTrue(loss2[0] > loss2[-1])
        
        # Check execution time
        start_time = time.time()
        model.fit(batch1[0], batch1[1], epochs=1, verbose=0)
        end_time = time.time()
        execution_time = end_time - start_time
        self.assertTrue(execution_time < 1.0)  # Assuming it should take less than 1 second

    def test_model_save_and_load(self):
        model = ContentBasedRecommender(input_dim=10, output_dim=5)
        model.fit(self.X_train, self.y_train, epochs=5)
        
        # Save the model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, 'test_model')
            model.save(model_path)
            
            # Load the model
            loaded_model = ContentBasedRecommender.load(model_path)
            
            # Compare predictions
            original_predictions = model.predict(self.X_test)
            loaded_predictions = loaded_model.predict(self.X_test)
            np.testing.assert_allclose(original_predictions, loaded_predictions, rtol=1e-5, atol=1e-5)

    def test_model_serving_interface(self):
        model = ContentBasedRecommender(input_dim=10, output_dim=5)
        model.fit(self.X_train, self.y_train, epochs=5)
        
        # Test with raw input
        raw_input = np.random.rand(1, 10).tolist()
        result = model.serve(raw_input)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(isinstance(x, float) for x in result))
        
        # Test with expected output
        expected_output = np.random.rand(5).tolist()
        similarity = model.compute_similarity(raw_input, expected_output)
        
        self.assertIsInstance(similarity, float)
        self.assertTrue(0 <= similarity <= 1)

if __name__ == '__main__':
    unittest.main()