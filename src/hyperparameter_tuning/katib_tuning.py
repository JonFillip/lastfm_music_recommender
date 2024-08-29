from kubeflow.katib import KatibClient
import yaml
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('hyperparameter_tuning')

def load_config():
    with open('configs/pipeline_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_experiment(name, namespace):
    config = load_config()
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
                "type": "minimize",
                "goal": 0.01,
                "objectiveMetricName": hp_config['objective']
            },
            "algorithm": {
                "algorithmName": hp_config['algorithm']
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
                    "name": "learning_rate",
                    "parameterType": "double",
                    "feasibleSpace": {
                        "min": str(hp_config['parameters']['learning_rate']['min']),
                        "max": str(hp_config['parameters']['learning_rate']['max'])
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
                        "name": "learning_rate",
                        "description": "Learning rate for optimizer",
                        "reference": "learning_rate"
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
                                        "image": "your-docker-image:latest",
                                        "command": [
                                            "python",
                                            "/app/train.py",
                                            "--hidden_layers=${trialParameters.hidden_layers}",
                                            "--neurons=${trialParameters.neurons}",
                                            "--learning_rate=${trialParameters.learning_rate}"
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

def run_hyperparameter_tuning():
    try:
        log_step(logger, 'Starting Hyperparameter Tuning', 'Katib')
        
        katib_client = KatibClient()
        
        experiment = create_experiment("music-recommender-tuning", "default")
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

def main():
    try:
        results = run_hyperparameter_tuning()
        # You can save the results or pass them to the next step in your pipeline
        return results
    except Exception as e:
        log_error(logger, e, 'Hyperparameter Tuning Main')
        raise

if __name__ == '__main__':
    main()