from scipy import stats
from google.cloud import monitoring_v3, storage, bigquery
from google.api import label_pb2 as ga_label
from google.api import metric_pb2 as ga_metric
from google.protobuf import duration_pb2 as duration
from src.data_processing.data_validation import generate_schema, validate_data, load_config
import yaml
import tensorflow_data_validation as tfdv
import argparse
import json
import pandas as pd
import datetime
import random



def setup_vertex_ai_monitoring(project_id, model_name):
    """Sets up custom metrics for Vertex AI monitoring in Cloud Monitoring."""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    # Define metrics
    metrics = [
        {"type": f"custom.googleapis.com/vertex_ai/{model_name}/prediction_drift",
        "kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
        "value_type": ga_metric.MetricDescriptor.ValueType.DOUBLE,
        "description": "Prediction drift metric for Vertex AI model"},
        {"type": f"custom.googleapis.com/vertex_ai/{model_name}/data_drift",
        "kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
        "value_type": ga_metric.MetricDescriptor.ValueType.DOUBLE,
        "description": "Data drift metric for Vertex AI model"},
        {"type": f"custom.googleapis.com/vertex_ai/{model_name}/prediction_latency",
        "kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
        "value_type": ga_metric.MetricDescriptor.ValueType.INT64,
        "description": "Latency of prediction requests in milliseconds"},
    ]

    for metric in metrics:
        descriptor = ga_metric.MetricDescriptor()
        descriptor.type = metric["type"]
        descriptor.metric_kind = metric["kind"]
        descriptor.value_type = metric["value_type"]
        descriptor.description = metric["description"]

        descriptor = client.create_metric_descriptor(
            name=project_name,
            metric_descriptor=descriptor
        )
        print(f"Created {descriptor.name}")

def create_data_drift_alert(project_id, model_name):
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    alert_policy = {
        "display_name": f"{model_name} Data Drift Alert",
        "conditions": [{
            "display_name": "Data drift exceeds threshold",
            "condition_threshold": {
                "filter": f'metric.type="custom.googleapis.com/vertex_ai/{model_name}/data_drift"',
                "comparison": monitoring_v3.ComparisonType.COMPARISON_GT,
                "threshold_value": 0.1,
                "duration": duration.Duration(seconds=300)
            }
        }],
        "notification_channels": [f"projects/{project_id}/notificationChannels/your-channel-id"],
        "combiner": monitoring_v3.AlertPolicy.Combiner.OR,
    }

    policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )
    print(f"Created alert policy: {policy.name}")

def create_prediction_drift_alert(project_id, model_name):
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    alert_policy = {
        "display_name": f"{model_name} Prediction Drift Alert",
        "conditions": [{
            "display_name": "Prediction drift exceeds threshold",
            "condition_threshold": {
                "filter": f'metric.type="custom.googleapis.com/vertex_ai/{model_name}/prediction_drift"',
                "comparison": monitoring_v3.ComparisonType.COMPARISON_GT,
                "threshold_value": 0.1,
                "duration": duration.Duration(seconds=300)
            }
        }],
        "notification_channels": [f"projects/{project_id}/notificationChannels/your-channel-id"],
        "combiner": monitoring_v3.AlertPolicy.Combiner.OR,
    }

    policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )
    print(f"Created alert policy: {policy.name}")

def create_resource_utilization_alert(project_id, model_name):
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    alert_policy = {
        "display_name": f"{model_name} Resource Utilization Alert",
        "conditions": [
            {
                "display_name": "High CPU utilization",
                "condition_threshold": {
                    "filter": 'metric.type="compute.googleapis.com/instance/cpu/utilization"',
                    "comparison": monitoring_v3.ComparisonType.COMPARISON_GT,
                    "threshold_value": 0.8,
                    "duration": duration.Duration(seconds=300)
                }
            },
            {
                "display_name": "High memory utilization",
                "condition_threshold": {
                    "filter": 'metric.type="compute.googleapis.com/instance/memory/utilization"',
                    "comparison": monitoring_v3.ComparisonType.COMPARISON_GT,
                    "threshold_value": 0.8,
                    "duration": duration.Duration(seconds=300)
                }
            },
            {
                "display_name": "High GPU utilization",
                "condition_threshold": {
                    "filter": 'metric.type="compute.googleapis.com/instance/gpu/utilization"',
                    "comparison": monitoring_v3.ComparisonType.COMPARISON_GT,
                    "threshold_value": 0.8,
                    "duration": duration.Duration(seconds=300)
                }
            }
        ],
        "notification_channels": [f"projects/{project_id}/notificationChannels/your-channel-id"],
        "combiner": monitoring_v3.AlertPolicy.Combiner.OR,
    }

    policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )
    print(f"Created alert policy: {policy.name}")

def create_latency_alert(project_id, model_name):
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    alert_policy = {
        "display_name": f"{model_name} Prediction Latency Alert",
        "conditions": [{
            "display_name": "High prediction latency",
            "condition_threshold": {
                "filter": f'metric.type="custom.googleapis.com/vertex_ai/{model_name}/prediction_latency"',
                "comparison": monitoring_v3.ComparisonType.COMPARISON_GT,
                "threshold_value": 1000,  # 1000 ms
                "duration": duration.Duration(seconds=60)
            }
        }],
        "notification_channels": [f"projects/{project_id}/notificationChannels/your-channel-id"],
        "combiner": monitoring_v3.AlertPolicy.Combiner.OR,
    }

    policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )
    print(f"Created alert policy: {policy.name}")

def log_request_response(project_id, model_name, request, response, latency_ms, sampling_rate=0.1):
    """
    Logs serving request/response data and latency to Cloud Storage with optional sampling.
    
    Args:
        project_id (str): GCP project ID
        model_name (str): Name of the Vertex AI model
        request (dict): Request data
        response (dict): Response data
        latency_ms (float): Latency of the request in milliseconds
        sampling_rate (float): Rate at which to sample logs (0.0 to 1.0, default 1.0)
    """
    if sampling_rate >= 1 or random.random() < sampling_rate:
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(f"{project_id}-vertex-ai-logs")
        blob = bucket.blob(f"{model_name}/logs/{datetime.datetime.now().isoformat()}.json")
        log_entry = {
            "request": request,
            "response": response,
            "latency_ms": latency_ms,
            "timestamp": datetime.datetime.now().isoformat()
        }
        blob.upload_from_string(json.dumps(log_entry))
        print(f"Logged request/response for {model_name} (latency: {latency_ms}ms)")

def check_existing_statistics_and_schema(project_id, model_name):
    bq_client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.model_monitoring.{model_name}_serving_stats"
    
    try:
        query = f"SELECT * FROM `{table_id}` ORDER BY timestamp DESC LIMIT 1"
        existing_stats = bq_client.query(query).result()
        existing_stats = list(existing_stats)[0] if existing_stats.total_rows > 0 else None
    except Exception as e:
        print(f"Error checking existing statistics: {e}")
        existing_stats = None

    config = load_config()
    schema_path = config['data_validation']['schema_path']
    
    try:
        schema = tfdv.load_schema_text(schema_path)
    except:
        schema = None

    return existing_stats, schema

def compute_and_store_statistics(project_id, model_name, existing_stats, existing_schema):
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(f"{project_id}-vertex-ai-logs")
    blobs = bucket.list_blobs(prefix=f"{model_name}/logs/")

    data = []
    for blob in blobs:
        content = json.loads(blob.download_as_string())
        data.append(content)

    df = pd.DataFrame(data)
    
    if existing_schema is None:
        schema = generate_schema(df)
    else:
        schema = existing_schema
    
    stats, anomalies = validate_data(df, schema)
    
    bq_client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.model_monitoring.{model_name}_serving_stats"
    
    row_to_insert = {
        "timestamp": datetime.datetime.now().isoformat(),
        "statistics": json.dumps(stats),
        "anomalies": json.dumps(anomalies)
    }

    errors = bq_client.insert_rows_json(table_id, [row_to_insert])
    if errors:
        print(f"Encountered errors while inserting rows: {errors}")
    else:
        print("New statistics added to BigQuery")

    return stats, anomalies

def detect_data_drift(project_id, model_name, current_stats, baseline_stats):
    if baseline_stats is None:
        print("No baseline statistics available. Unable to detect drift.")
        return

    # Implement your data drift detection logic here
    # This is a simplified example
    drift_score = stats.ks_2samp(current_stats['feature1'], baseline_stats['feature1']).statistic

    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/vertex_ai/{model_name}/data_drift"
    series.resource.type = "aiplatform.googleapis.com/Endpoint"
    series.resource.labels["model_name"] = model_name
    point = series.points.add()
    point.value.double_value = drift_score
    now = datetime.datetime.now()
    point.interval.end_time.seconds = int(now.timestamp())
    point.interval.end_time.nanos = int((now.timestamp() - int(now.timestamp())) * 10**9)
    client.create_time_series(name=project_name, time_series=[series])

    print(f"Data drift score: {drift_score}")

def detect_prediction_drift(project_id, model_name, current_stats, baseline_stats):
    if baseline_stats is None:
        print("No baseline statistics available. Unable to detect prediction drift.")
        return

    # Implement your prediction drift detection logic here
    # This is a simplified example
    drift_score = abs(current_stats['prediction_mean'] - baseline_stats['prediction_mean']) / baseline_stats['prediction_std']

    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/vertex_ai/{model_name}/prediction_drift"
    series.resource.type = "aiplatform.googleapis.com/Endpoint"
    series.resource.labels["model_name"] = model_name
    point = series.points.add()
    point.value.double_value = drift_score
    now = datetime.datetime.now()
    point.interval.end_time.seconds = int(now.timestamp())
    point.interval.end_time.nanos = int((now.timestamp() - int(now.timestamp())) * 10**9)
    client.create_time_series(name=project_name, time_series=[series])

    print(f"Prediction drift score: {drift_score}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup Vertex AI monitoring')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model_name', required=True, help='Vertex AI model name')
    parser.add_argument('--sampling_rate', type=float, default=1.0, help='Sampling rate for request/response logging')
    args = parser.parse_args()

    setup_vertex_ai_monitoring(args.project_id, args.model_name)
    create_data_drift_alert(args.project_id, args.model_name)
    create_prediction_drift_alert(args.project_id, args.model_name)
    create_resource_utilization_alert(args.project_id, args.model_name)
    create_latency_alert(args.project_id, args.model_name)

    # Check for existing statistics and schema
    existing_stats, existing_schema = check_existing_statistics_and_schema(args.project_id, args.model_name)

    # Compute and store current statistics
    current_stats, anomalies = compute_and_store_statistics(args.project_id, args.model_name, existing_stats, existing_schema)

    # Detect data drift if baseline statistics exist
    if existing_stats:
        detect_data_drift(args.project_id, args.model_name, current_stats, existing_stats)
        detect_prediction_drift(args.project_id, args.model_name, current_stats, existing_stats)
    else:
        print("No existing statistics found. Current statistics will be used as the baseline for future comparisons.")

    print("Vertex AI monitoring setup and checks completed successfully!")