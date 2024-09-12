from datetime import time
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logging_utils import setup_logger, log_error, log_metric, log_step
from deployment.vertex_ai.vertex_ai_monitoring import create_accuracy_degradation_alert
import yaml
import json
from typing import Dict, Any
from google.cloud import monitoring_v3

logger = setup_logger('model_evaluation')

def load_config():
    with open('configs/pipeline_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_metrics(metrics_path):
    with open(metrics_path, 'r') as f:
        return json.load(f)

def mean_average_precision(y_true, y_pred, k=10):
    """Calculate Mean Average Precision@K"""
    return np.mean([average_precision(yt, yp, k) for yt, yp in zip(y_true, y_pred)])

def average_precision(y_true, y_pred, k=10):
    """Calculate Average Precision@K for a single sample"""
    if len(y_pred) > k:
        y_pred = y_pred[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(y_true), k)

def diversity(recommendations):
    """Calculate diversity of recommendations"""
    unique_items = set()
    for rec in recommendations:
        unique_items.update(rec)
    return len(unique_items) / (len(recommendations) * len(recommendations[0]))

def novelty(recommendations, popularity):
    """Calculate novelty of recommendations"""
    return np.mean([np.mean([- np.log2(popularity.get(i, 0.01)) for i in rec]) for rec in recommendations])

def evaluate_model(model, X_test, y_test, metrics_path, item_popularity):
    try:
        config = load_config()
        log_step(logger, 'Model Evaluation', 'Evaluation')
        
        # Load saved metrics
        saved_metrics = load_metrics(metrics_path)
        
        # Evaluate model on test set
        y_pred = model.predict(X_test)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate additional metrics
        y_pred_binary = (y_pred > 0.5).astype(int)
        precision = precision_score(y_test, y_pred_binary, average='weighted')
        recall = recall_score(y_test, y_pred_binary, average='weighted')
        f1 = f1_score(y_test, y_pred_binary, average='weighted')
        ndcg = ndcg_score(y_test, y_pred)
        map_score = mean_average_precision(y_test, y_pred)
        diversity_score = diversity(y_pred)
        novelty_score = novelty(y_pred, item_popularity)
        
        # Combine saved metrics and new evaluations
        results = {
            **saved_metrics,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1,
            "test_ndcg": ndcg,
            "test_mean_average_precision": map_score,
            "test_diversity": diversity_score,
            "test_novelty": novelty_score
        }
        
        # Log all metrics
        for metric, value in results.items():
            log_metric(logger, metric, value, 'Model Evaluation')
        
        return results
    except Exception as e:
        log_error(logger, e, 'Model Evaluation')
        raise

def visualize_results(results: Dict[str, float], output_path: str):
    """Visualize evaluation results"""
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Model Evaluation Results')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def update_custom_metrics(project_id: str, model_name: str, metrics: Dict[str, float]):
    """Update custom metrics in Vertex AI"""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    for metric_name, value in metrics.items():
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/vertex_ai/{model_name}/{metric_name}"
        series.resource.type = "vertex_ai_model"
        series.resource.labels["model_name"] = model_name
        point = series.points.add()
        point.value.double_value = value
        now = time.time()
        point.interval.end_time.seconds = int(now)
        point.interval.end_time.nanos = int((now - int(now)) * 10**9)
        client.create_time_series(name=project_name, time_series=[series])

    logger.info(f"Updated custom metrics for model {model_name}")

def main(model_path: str, test_data_path: str, metrics_path: str, output_path: str, project_id: str, model_name: str):
    try:
        config = load_config()
        model = tf.keras.models.load_model(model_path)
        
        # Load test data
        if test_data_path.endswith('.npz'):
            test_data = np.load(test_data_path)
            X_test, y_test = test_data['X'], test_data['y']
        elif test_data_path.endswith('.csv'):
            test_data = pd.read_csv(test_data_path)
            X_test, y_test = test_data.drop('target', axis=1), test_data['target']
        else:
            raise ValueError(f"Unsupported data format: {test_data_path}")
        
        # Load item popularity (assume it's available)
        with open('data/item_popularity.json', 'r') as f:
            item_popularity = json.load(f)
        
        # Perform evaluation
        evaluation_results = evaluate_model(model, X_test, y_test, metrics_path, item_popularity)
        
        # Visualize results
        visualize_results(evaluation_results, f"{output_path}/evaluation_results.png")
        
        # Save results
        with open(f"{output_path}/evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Update custom metrics in Vertex AI
        update_custom_metrics(project_id, model_name, evaluation_results)
        
        # Set up or update monitoring alert
        create_accuracy_degradation_alert(
            project_id=project_id,
            model_name=model_name,
            absolute_threshold=0.8,  # Set your desired threshold
            degradation_rate_threshold=0.05,  # Set your desired degradation rate
            time_window_seconds=86400  # 24 hours
        )
        
        logger.info("Evaluation completed, results saved, and monitoring alert updated.")
        return evaluation_results
    except Exception as e:
        log_error(logger, e, 'Model Evaluation Main')
        raise

if __name__ == '__main__':
    config = load_config()
    main(config['model_path'], config['test_data_path'], config['metrics_path'], config['output_path'], 
        config['project_id'], config['model_name'])