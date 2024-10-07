import unittest
from unittest.mock import patch, MagicMock
import yaml
from src.hyperparameter_tuning.katib_tuning import (
    load_config,
    create_experiment,
    run_hyperparameter_tuning,
    main
)

class TestHyperparameterTuning(unittest.TestCase):

    def setUp(self):
        self.mock_config = {
            'hyperparameter_tuning': {
                'max_trials': 10,
                'parameters': {
                    'hidden_layers': {'min': 1, 'max': 5},
                    'neurons': {'min': 32, 'max': 256},
                    'learning_rate': {'min': 0.0001, 'max': 0.1}
                }
            }
        }

    @patch('src.hyperparameter_tuning.katib_tuning.open')
    def test_load_config(self, mock_open):
        mock_open.return_value.__enter__.return_value = MagicMock()
        mock_yaml_load = MagicMock(return_value=self.mock_config)
        with patch('src.hyperparameter_tuning.katib_tuning.yaml.safe_load', mock_yaml_load):
            config = load_config()
        self.assertEqual(config, self.mock_config)

    @patch('src.hyperparameter_tuning.katib_tuning.load_config')
    def test_create_experiment(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        experiment = create_experiment("test-experiment", "default", "train_data.csv", "val_data.csv")
        
        self.assertEqual(experiment['metadata']['name'], "test-experiment")
        self.assertEqual(experiment['metadata']['namespace'], "default")
        self.assertEqual(experiment['spec']['maxTrialCount'], 10)
        self.assertIn('parameters', experiment['spec'])
        self.assertEqual(len(experiment['spec']['parameters']), 6)

    @patch('src.hyperparameter_tuning.katib_tuning.KatibClient')
    @patch('src.hyperparameter_tuning.katib_tuning.create_experiment')
    def test_run_hyperparameter_tuning(self, mock_create_experiment, mock_katib_client):
        mock_client = MagicMock()
        mock_katib_client.return_value = mock_client
        mock_create_experiment.return_value = {"metadata": {"name": "test-experiment"}}
        mock_client.get_optimal_hyperparameters.return_value = {"bestTrialName": "test-trial"}

        results = run_hyperparameter_tuning("train_data.csv", "val_data.csv")

        mock_create_experiment.assert_called_once()
        mock_client.create_experiment.assert_called_once()
        mock_client.wait_for_experiment.assert_called_once()
        mock_client.get_optimal_hyperparameters.assert_called_once()
        self.assertEqual(results, {"bestTrialName": "test-trial"})

    @patch('src.hyperparameter_tuning.katib_tuning.run_hyperparameter_tuning')
    def test_main(self, mock_run_hyperparameter_tuning):
        mock_results = {"bestTrialName": "test-trial"}
        mock_run_hyperparameter_tuning.return_value = mock_results

        results = main("train_data.csv", "val_data.csv")

        mock_run_hyperparameter_tuning.assert_called_once_with("train_data.csv", "val_data.csv")
        self.assertEqual(results, mock_results)

if __name__ == '__main__':
    unittest.main()