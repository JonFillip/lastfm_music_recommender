import unittest
from unittest.mock import patch, MagicMock
from src.hyperparameter_tuning.katib_tuning import create_experiment, run_hyperparameter_tuning

class TestHyperparameterTuning(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            'hyperparameter_tuning': {
                'max_trials': 10,
                'parameters': {
                    'hidden_layers': {'min': 1, 'max': 3},
                    'neurons': {'min': 32, 'max': 256},
                    'learning_rate': {'min': 0.0001, 'max': 0.1}
                }
            }
        }

    def test_create_experiment(self):
        experiment = create_experiment('test-experiment', 'default', 'train_data.csv', 'val_data.csv', self.mock_config)
        
        self.assertEqual(experiment['metadata']['name'], 'test-experiment')
        self.assertEqual(experiment['metadata']['namespace'], 'default')
        self.assertEqual(experiment['spec']['maxTrialCount'], 10)
        
        parameters = experiment['spec']['parameters']
        self.assertEqual(len(parameters), 6)  # hidden_layers, neurons, embedding_dim, learning_rate, batch_size, dropout_rate
        
        hidden_layers_param = next(p for p in parameters if p['name'] == 'hidden_layers')
        self.assertEqual(hidden_layers_param['feasibleSpace']['min'], '1')
        self.assertEqual(hidden_layers_param['feasibleSpace']['max'], '3')

    @patch('src.hyperparameter_tuning.katib_tuning.KatibClient')
    def test_run_hyperparameter_tuning(self, mock_katib_client):
        mock_client = MagicMock()
        mock_katib_client.return_value = mock_client
        
        mock_client.get_optimal_hyperparameters.return_value = {
            'currentOptimalTrial': {
                'parameterAssignments': [
                    {'name': 'hidden_layers', 'value': '2'},
                    {'name': 'neurons', 'value': '128'},
                    {'name': 'learning_rate', 'value': '0.001'}
                ]
            }
        }
        
        results = run_hyperparameter_tuning('train_data.csv', 'val_data.csv', 'config.yaml')
        
        mock_katib_client.assert_called_once()
        mock_client.create_experiment.assert_called_once()
        mock_client.wait_for_experiment.assert_called_once()
        mock_client.get_optimal_hyperparameters.assert_called_once()
        
        self.assertIn('currentOptimalTrial', results)
        self.assertIn('parameterAssignments', results['currentOptimalTrial'])

if __name__ == '__main__':
    unittest.main()