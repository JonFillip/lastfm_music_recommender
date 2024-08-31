import argparse
from kubeflow.katib import KatibClient
import yaml
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('hyperparameter_tuning')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_experiment(name, namespace, train_data, val_data, config):
    hp_config = config['hyperparameter_tuning']

    experiment = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": name,
            "namespace": namespace
        },
        "spec": {
            "objective": {
                "type": "maximize",
                "goal": 0.9,
                "objectiveMetricName": "val_cosine_similarity"
            },
            "algorithm": {
                "algorithmName": "random"
            },
            "maxTrialCount": hp_config['max_trials'],
            "maxFailedTrialCount": 3,
            "parallelTrialCount": 2,
            "parameters": [
                {
                    "name": "hidden_layers",
                    "parameterType": "int",
                    "feasibleSpace": {
                        "min": str(hp_config['parameters']['hidden_layers']['min']),
                        "max": str(hp_config['parameters']['hidden_layers']['max'])
                    }
                },
                {
                    "name": "neurons",
                    "parameterType": "int",
                    "feasibleSpace": {
                        "min": str(hp_config['parameters']['neurons']['min']),
                        "max": str(hp_config['parameters']['neurons']['max'])
                    }
                },
                {
                    "name": "embedding_dim",
                    "parameterType": "int",
                    "feasibleSpace": {
                        "min": "32",
                        "max": "256"
                    }
                },
                {
                    "name": "learning_rate",
                    "parameterType": "double",
                    "feasibleSpace": {
                        "min": str(hp_config['parameters']['learning_rate']['min']),
                        "max": str(hp_config['parameters']['learning_rate']['max'])
                    }
                },
                {
                    "name": "batch_size",
                    "parameterType": "int",
                    "feasibleSpace": {
                        "min": "32",
                        "max": "256"
                    }
                },
                {
                    "name": "dropout_rate",
                    "parameterType": "double",
                    "feasibleSpace": {
                        "min": "0.1",
                        "max": "0.5"
                    }
                }
            ],
            "trialTemplate": {
                "primaryContainerName": "training",
                "trialParameters": [
                    {
                        "name": "hidden_layers",
                        "description": "Number of hidden layers",
                        "reference": "hidden_layers"
                    },
                    {
                        "name": "neurons",
                        "description": "Number of neurons per hidden layer",
                        "reference": "neurons"
                    },
                    {
                        "name": "embedding_dim",
                        "description": "Dimension of the embedding layer",
                        "reference": "embedding_dim"
                    },
                    {
                        "name": "learning_rate",
                        "description": "Learning rate for optimizer",
                        "reference": "learning_rate"
                    },
                    {
                        "name": "batch_size",
                        "description": "Batch size for training",
                        "reference": "batch_size"
                    },
                    {
                        "name": "dropout_rate",
                        "description": "Dropout rate for regularization",
                        "reference": "dropout_rate"
                    }
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "training",
                                        "image": "gcr.io/your-project-id/lastfm-music-recommender:latest",
                                        "command": [
                                            "python",
                                            "/app/src/algorithms/content_based.py",
                                            "--train_data", train_data,
                                            "--val_data", val_data,
                                            "--hidden_layers", "${trialParameters.hidden_layers}",
                                            "--neurons", "${trialParameters.neurons}",
                                            "--embedding_dim", "${trialParameters.embedding_dim}",
                                            "--learning_rate", "${trialParameters.learning_rate}",
                                            "--batch_size", "${trialParameters.batch_size}",
                                            "--dropout_rate", "${trialParameters.dropout_rate}"
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
    }
    return experiment

def run_hyperparameter_tuning(train_data, val_data, config_path):
    try:
        log_step(logger, 'Starting Hyperparameter Tuning', 'Katib')
        
        config = load_config(config_path)
        katib_client = KatibClient()
        
        experiment = create_experiment("music-recommender-tuning", "default", train_data, val_data, config)
        katib_client.create_experiment(experiment)
        
        logger.info("Hyperparameter tuning experiment created successfully")
        
        # Wait for the experiment to complete
        katib_client.wait_for_experiment("music-recommender-tuning", "default")
        
        # Get the results
        results = katib_client.get_optimal_hyperparameters("music-recommender-tuning", "default")
        
        logger.info(f"Optimal hyperparameters: {results}")
        return results
    except Exception as e:
        log_error(logger, e, 'Hyperparameter Tuning')
        raise

def main(train_data, val_data, config_path):
    try:
        results = run_hyperparameter_tuning(train_data, val_data, config_path)
        # You can save the results or pass them to the next step in your pipeline
        return results
    except Exception as e:
        log_error(logger, e, 'Hyperparameter Tuning Main')
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for content-based recommender')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--config_path', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.train_data, args.val_data, args.config_path)