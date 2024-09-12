from scipy import stats
from scipy.stats import ks_2samp
from google.cloud import monitoring_v3, storage, bigquery, aiplatform
from google.api import label_pb2 as ga_label
from google.api import metric_pb2 as ga_metric
from google.protobuf import duration_pb2 as duration
from src.data_processing.data_validation import generate_schema, validate_data, load_config, load_statistics_from_gcs, load_schema_from_gcs, compare_statistics, compare_schemas
import yaml
import tensorflow_data_validation as tfdv
import argparse
import json
import pandas as pd
import datetime
import random
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('vertex_ai_pipeline_monitoring')

def setup_vertex_ai_monitoring(project_id, model_name):
    """Sets up custom metrics for Vertex AI monitoring in Cloud Monitoring."""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    # Define metrics
    metrics = [
        {
            "type": f"custom.googleapis.com/vertex_ai/{model_name}/prediction_drift",
            "kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
            "value_type": ga_metric.MetricDescriptor.ValueType.DOUBLE,
            "description": "Prediction drift metric for Vertex AI model"
        },
        {
            "type": f"custom.googleapis.com/vertex_ai/{model_name}/data_drift",
            "kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
            "value_type": ga_metric.MetricDescriptor.ValueType.DOUBLE,
            "description": "Data drift metric for Vertex AI model"
        },
        {
            "type": f"custom.googleapis.com/vertex_ai/{model_name}/prediction_latency",
            "kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
            "value_type": ga_metric.MetricDescriptor.ValueType.INT64,
            "description": "Latency of prediction requests in milliseconds"
        },
        {
            "type": f"custom.googleapis.com/vertex_ai/{model_name}/accuracy",
            "kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
            "value_type": ga_metric.MetricDescriptor.ValueType.DOUBLE,
            "description": "Accuracy of the Vertex AI model"
        },
        {
            "type": f"custom.googleapis.com/vertex_ai/{model_name}/schema_drift",
            "kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
            "value_type": ga_metric.MetricDescriptor.ValueType.INT64,  # 0 or 1 indicating schema drift
            "description": "Schema drift metric for Vertex AI model"
        },
        {
            "type": f"custom.googleapis.com/vertex_ai/{model_name}/missing_schema",
            "kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
            "value_type": ga_metric.MetricDescriptor.ValueType.INT64,  # 0 or 1 indicating missing schema
            "description": "Indicates whether the baseline schema is missing"
        },
        {
            "type": f"custom.googleapis.com/vertex_ai/{model_name}/missing_statistics",
            "kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
            "value_type": ga_metric.MetricDescriptor.ValueType.INT64,  # 0 or 1 indicating missing schema
            "description": "Indicates whether the baseline statistics is missing"
        }
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


def create_accuracy_degradation_alert(project_id: str, model_name: str, absolute_threshold: float, degradation_rate_threshold: float, time_window_seconds: int = 86400):
    """
    Create an alert in Google Cloud Monitoring for model accuracy degradation.
    The alert will trigger if:
    1. The accuracy falls below an absolute threshold.
    2. The accuracy degrades by a certain rate within a specified time window.
    
    :param project_id: GCP Project ID
    :param model_name: Name of the model
    :param absolute_threshold: The absolute accuracy threshold to trigger an alert (e.g., accuracy < 0.85).
    :param degradation_rate_threshold: The degradation rate threshold over time (e.g., 0.05 for a 5% drop).
    :param time_window_seconds: The time window to monitor for accuracy degradation (default is 24 hours).
    """
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    # Condition 1: Absolute accuracy degradation
    absolute_condition = {
        "display_name": "Accuracy below absolute threshold",
        "condition_threshold": {
            "filter": f'metric.type="custom.googleapis.com/vertex_ai/{model_name}/accuracy"',
            "comparison": monitoring_v3.ComparisonType.COMPARISON_LT,
            "threshold_value": absolute_threshold,
            "duration": {"seconds": 300},  # Trigger if accuracy stays below threshold for 5 minutes
            "aggregations": [{
                "alignment_period": {"seconds": 300},
                "per_series_aligner": monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
            }]
        }
    }

    # Condition 2: Degradation rate over time
    degradation_condition = {
        "display_name": "Accuracy degradation over time",
        "condition_threshold": {
            "filter": f'metric.type="custom.googleapis.com/vertex_ai/{model_name}/accuracy"',
            "comparison": monitoring_v3.ComparisonType.COMPARISON_LT,
            "threshold_value": degradation_rate_threshold,  # Set degradation rate threshold
            "duration": {"seconds": time_window_seconds},  # Time window (e.g., 24 hours)
            "aggregations": [{
                "alignment_period": {"seconds": time_window_seconds},
                "per_series_aligner": monitoring_v3.Aggregation.Aligner.ALIGN_DELTA,
            }]
        }
    }

    # Create the alert policy
    alert_policy = {
        "display_name": f"Accuracy Degradation Alert for {model_name}",
        "conditions": [absolute_condition, degradation_condition],
        "notification_channels": [f"projects/{project_id}/notificationChannels/your-channel-id"],  # Replace with actual channel
        "combiner": monitoring_v3.AlertPolicy.Combiner.OR,  # Trigger if either condition is met
        "enabled": True
    }

    # Apply the policy to the project
    policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )

    logger.info(f"Created accuracy degradation alert policy: {policy.name}")
    return policy


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

def create_schema_drift_alert(project_id: str, model_name: str):
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    # Create the alert policy definition
    alert_policy = monitoring_v3.AlertPolicy(
        display_name=f"{model_name} Schema Drift Alert",
        conditions=[{
            "display_name": "Schema Drift Detected",
            "condition_threshold": {
                "filter": f'metric.type="custom.googleapis.com/vertex_ai/{model_name}/schema_drift"',
                "comparison": monitoring_v3.ComparisonType.COMPARISON_GT,
                "threshold_value": 1,
                "duration": duration.Duration(seconds=300)  # Set duration for continuous drift
            }
        }],
        notification_channels=[f"projects/{project_id}/notificationChannels/your-channel-id"],  # Replace with actual channel ID
        combiner=monitoring_v3.AlertPolicy.Combiner.OR,  # How to combine multiple conditions
        enabled=True
    )

    # Apply the alert policy
    policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )

    print(f"Schema drift alert policy created: {policy.name}")

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

def load_baseline_stats_and_schema(bucket_name, model_name, schema_version):
    """Load baseline statistics and schema from Google Cloud Storage."""
    # Load baseline statistics from GCS
    baseline_stats = load_statistics_from_gcs(bucket_name, model_name, data_type='train')

    # Load schema from GCS
    schema = load_schema_from_gcs(bucket_name, model_name, schema_version)

    return baseline_stats, schema


def handle_missing_statistics(project_id, stat_type, model_name):
    """
    Handles the case where statistics are missing.
    Logs a warning and optionally triggers alerts for missing statistics.
    """
    warning_msg = f"Missing {stat_type} statistics for model: {model_name}. Skipping drift detection."
    logger.warning(warning_msg)

    # Optionally, log missing statistics as a custom metric in Google Cloud Monitoring
    # This helps track the issue and potentially trigger alerts.
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/vertex_ai/{model_name}/missing_statistics"
    series.resource.type = "aiplatform.googleapis.com/Endpoint"
    series.resource.labels["model_name"] = model_name
    point = series.points.add()
    point.value.double_value = 1  # Use '1' to indicate missing stats
    now = datetime.datetime.now()
    point.interval.end_time.seconds = int(now.timestamp())
    point.interval.end_time.nanos = int((now.timestamp() - int(now.timestamp())) * 10**9)
    client.create_time_series(name=project_name, time_series=[series])

    logger.info(f"Logged missing statistics for model {model_name}")

def handle_missing_schema(project_id, model_name):
    """
    Handles the case where the schema is missing.
    Logs a warning and optionally triggers alerts for missing schema.
    """
    warning_msg = f"Missing schema for model: {model_name}. Skipping schema drift detection."
    logger.warning(warning_msg)

    # Optionally log missing schema as a custom metric in Google Cloud Monitoring
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/vertex_ai/{model_name}/missing_schema"
    series.resource.type = "aiplatform.googleapis.com/Endpoint"
    series.resource.labels["model_name"] = model_name
    point = series.points.add()
    point.value.double_value = 1  # '1' indicates missing schema
    now = datetime.datetime.now()
    point.interval.end_time.seconds = int(now.timestamp())
    point.interval.end_time.nanos = int((now.timestamp() - int(now.timestamp())) * 10**9)
    client.create_time_series(name=project_name, time_series=[series])

    logger.info(f"Logged missing schema for model {model_name} in Google Cloud Monitoring.")


def detect_data_drift(project_id, model_name, bucket_name, schema_version, drift_threshold):
    """
    Detects data drift by comparing current (serving) statistics with baseline (training) statistics.
    Logs the drift score directly to Google Cloud Monitoring.
    Returns the drift score (or None if statistics are missing).
    """
    try:
        log_step(logger, 'Detecting Data Drift', 'Data Drift Detection')

        # Load baseline statistics and schema
        log_step(logger, 'Loading Baseline Statistics and Schema', 'Data Drift Detection')
        today = datetime.datetime.now().strftime("%Y%m%d")
        
        # Load baseline statistics and check if they exist
        baseline_stats = load_statistics_from_gcs(bucket_name, model_name, 'train', today)
        if not baseline_stats:
            handle_missing_statistics(project_id, 'baseline', model_name)
            return None
        
        # Load serving statistics
        serving_stats = load_statistics_from_gcs(bucket_name, model_name, 'serving', today)
        if not serving_stats:
            handle_missing_statistics(project_id, 'serving', model_name)
            return None
        
        # Load schema from GCS
        schema = load_schema_from_gcs(bucket_name, model_name, schema_version)
        if not schema:
            logger.warning(f"No schema found for {model_name}. Skipping data drift detection.")
            return None

        # Compare statistics and check for anomalies
        log_step(logger, 'Comparing Statistics', 'Data Drift Detection')
        anomalies = compare_statistics(baseline_stats, serving_stats, schema)

        # Calculate and return the drift score
        drift_score = 0  # Initialize drift score
        significant_drift_detected = False

        for feature, anomaly in anomalies.anomaly_info.items():
            if anomaly:
                logger.warning(f"Data drift detected for feature {feature}: {anomaly.description}")
                drift_score += anomaly.severity

                # Check if drift score exceeds the threshold
                if drift_score > drift_threshold:
                    significant_drift_detected = True

                # Log drift score for the specific feature to Vertex AI Monitoring
                client = monitoring_v3.MetricServiceClient()
                project_name = f"projects/{project_id}"

                series = monitoring_v3.TimeSeries()
                series.metric.type = f"custom.googleapis.com/vertex_ai/{model_name}/data_drift"
                series.resource.type = "aiplatform.googleapis.com/Endpoint"
                series.resource.labels["model_name"] = model_name
                point = series.points.add()
                point.value.double_value = anomaly.severity
                now = datetime.datetime.now()
                point.interval.end_time.seconds = int(now.timestamp())
                point.interval.end_time.nanos = int((now.timestamp() - int(now.timestamp())) * 10**9)
                client.create_time_series(name=project_name, time_series=[series])

                logger.info(f"Logged data drift score for {feature}: {anomaly.severity}")

        # Log if significant drift is detected based on the threshold
        if significant_drift_detected:
            logger.warning(f"Significant data drift detected for {model_name}. Drift score: {drift_score} > {drift_threshold}")
        else:
            logger.info(f"No significant data drift detected for {model_name}. Drift score: {drift_score} <= {drift_threshold}")

        return drift_score

    except Exception as e:
        log_error(logger, e, 'Data Drift Detection')
        return None  # Return None if an error occurs


def detect_prediction_drift(project_id, model_name, bucket_name, drift_threshold):
    """
    Detects prediction drift using the Kolmogorov-Smirnov (KS) test and logs the drift score to Google Cloud Monitoring.
    Returns the drift score (or None if statistics are missing).
    """
    try:
        log_step(logger, 'Detecting Prediction Drift', 'Prediction Drift Detection')

        # Load training and serving statistics
        today = datetime.datetime.now().strftime("%Y%m%d")
        
        # Load baseline prediction statistics and check if they exist
        train_stats = load_statistics_from_gcs(bucket_name, model_name, 'train', today)
        if not train_stats:
            handle_missing_statistics(project_id, 'training', model_name)
            return None
        
        # Load serving statistics
        serving_stats = load_statistics_from_gcs(bucket_name, model_name, 'serving', today)
        if not serving_stats:
            handle_missing_statistics(project_id, 'serving', model_name)
            return None

        # Extract prediction buckets for KS test (assuming 'predictions' is the feature name)
        train_predictions = train_stats.datasets[0].features['similar_tracks'].num_stats.histograms[0].buckets
        serving_predictions = serving_stats.datasets[0].features['similar_tracks'].num_stats.histograms[0].buckets

        # Extract the counts from the buckets
        train_counts = [bucket.sample_count for bucket in train_predictions]
        serving_counts = [bucket.sample_count for bucket in serving_predictions]

        # Perform KS test
        statistic, p_value = ks_2samp(train_counts, serving_counts)

        # Log prediction drift score to Vertex AI Monitoring
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{project_id}"

        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/vertex_ai/{model_name}/prediction_drift"
        series.resource.type = "aiplatform.googleapis.com/Endpoint"
        series.resource.labels["model_name"] = model_name
        point = series.points.add()
        point.value.double_value = statistic  # Log the KS statistic
        now = datetime.datetime.now()
        point.interval.end_time.seconds = int(now.timestamp())
        point.interval.end_time.nanos = int((now.timestamp() - int(now.timestamp())) * 10**9)
        client.create_time_series(name=project_name, time_series=[series])

        logger.info(f"Logged prediction drift KS statistic: {statistic}")

        # Determine if drift is significant and return the drift score
        if statistic > drift_threshold:
            logger.warning(f"Prediction drift detected for model {model_name}: KS statistic = {statistic}")
        else:
            logger.info(f"No significant prediction drift detected for model {model_name}")

        return statistic

    except Exception as e:
        log_error(logger, e, 'Prediction Drift Detection')
        return None  # Return None if an error occurs


def monitor_traffic_split(project_id, endpoint_name):
    """Monitor the traffic split in Vertex AI to detect rollback."""
    try:
        log_step(logger, 'Monitoring traffic split', 'Rollback Monitoring')
        
        aiplatform.init(project=project_id)

        # Retrieve the endpoint
        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
        if not endpoints:
            log_error(logger, f"Endpoint {endpoint_name} not found.", "Rollback Monitoring")
            return None
        endpoint = endpoints[0]

        # Get the traffic split
        traffic_split = endpoint.traffic_split
        for model_id, traffic_percentage in traffic_split.items():
            logger.info(f"Model {model_id} is receiving {traffic_percentage}% of the traffic.")
        
        # Check if rollback happened (i.e., if traffic is no longer sent to the new model)
        if sum(traffic_split.values()) != 100:
            logger.warning("Traffic split does not sum to 100%, indicating a possible rollback.")
    
        return traffic_split
    except Exception as e:
        log_error(logger, e, "Rollback Monitoring")
        raise

def detect_schema_drift(project_id, model_name, bucket_name, schema_version):
    """
    Detects schema drift by comparing the current schema with the baseline (training) schema.
    Logs the drift to Google Cloud Monitoring if detected.
    Returns a boolean indicating whether schema drift was detected.
    """
    try:
        log_step(logger, 'Detecting Schema Drift', 'Schema Drift Detection')

        # Load baseline schema
        log_step(logger, 'Loading Baseline Schema', 'Schema Drift Detection')
        baseline_schema = load_schema_from_gcs(bucket_name, model_name, schema_version)
        if not baseline_schema:
            return handle_missing_schema(project_id, model_name)

        # Load current schema (Replace with your actual method of loading the current schema from serving data)
        log_step(logger, 'Loading Current Schema', 'Schema Drift Detection')
        current_schema = load_schema_from_gcs(bucket_name, model_name, 'serving_schema_version')  # Replace with actual logic for serving schema

        # Compare schemas and check for schema drift
        schema_drift_detected = compare_schemas(baseline_schema, current_schema)

        # Log schema drift to Google Cloud Monitoring if detected
        if schema_drift_detected:
            logger.info(f"Schema drift detected for model {model_name}.")
            client = monitoring_v3.MetricServiceClient()
            project_name = f"projects/{project_id}"

            series = monitoring_v3.TimeSeries()
            series.metric.type = f"custom.googleapis.com/vertex_ai/{model_name}/schema_drift"
            series.resource.type = "aiplatform.googleapis.com/Endpoint"
            series.resource.labels["model_name"] = model_name
            point = series.points.add()
            point.value.double_value = 1  # Use '1' to indicate schema drift
            now = datetime.datetime.now()
            point.interval.end_time.seconds = int(now.timestamp())
            point.interval.end_time.nanos = int((now.timestamp() - int(now.timestamp())) * 10**9)
            client.create_time_series(name=project_name, time_series=[series])

            logger.info(f"Logged schema drift for model {model_name} to Google Cloud Monitoring.")
        else:
            logger.info(f"No schema drift detected for model {model_name}.")

        return schema_drift_detected

    except Exception as e:
        log_error(logger, e, 'Schema Drift Detection')
        return None  # Return None if an error occurs

def create_rollback_alert(project_id, endpoint_name):
    """Create an alert in Google Cloud Monitoring for rollback events."""
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"
    
    alert_policy = {
        "display_name": f"Rollback Alert for {endpoint_name}",
        "conditions": [{
            "display_name": "Rollback detected",
            "condition_threshold": {
                "filter": f'metric.type="aiplatform.googleapis.com/Endpoint/traffic_split"',
                "comparison": monitoring_v3.ComparisonType.COMPARISON_LT,  # Define condition for rollback
                "threshold_value": 100,  # Set rollback condition here
                "duration": monitoring_v3.Duration(seconds=300)
            }
        }],
        "notification_channels": [f"projects/{project_id}/notificationChannels/your-channel-id"],  # Replace with your actual channel ID
        "combiner": monitoring_v3.AlertPolicy.Combiner.OR,
    }

    policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )
    logger.info(f"Created rollback alert policy: {policy.name}")

def monitor_and_log_rollbacks(project_id, endpoint_name):
    logger.info("Starting rollback monitoring...")
    traffic_split = monitor_traffic_split(project_id, endpoint_name)
    if traffic_split:
        create_rollback_alert(project_id, endpoint_name)

def trigger_retraining_pipeline(project_id: str, pipeline_name: str, gcs_input: str, model_name: str):
    """
    Trigger a Vertex AI pipeline for continuous retraining when performance degradation or drift is detected.
    """
    aiplatform.init(project=project_id)

    pipeline_params = {
        'input_data': gcs_input,
        'model_name': model_name
    }

    # Run the retraining pipeline
    pipeline_job = aiplatform.PipelineJob(
        display_name=f'Retraining - {model_name}',
        template_path=f'gs://{pipeline_name}',
        parameter_values=pipeline_params
    )

    pipeline_job.run()
    logger.info(f"Triggered retraining pipeline for {model_name}.")

    # Return the pipeline job ID for tracking
    return pipeline_job.resource_name

def setup_retraining_job_alert(project_id: str, notification_channel: str):
    """
    Set up a Cloud Monitoring alert for Vertex AI retraining jobs.
    This sends notifications whenever a new retraining job is created.
    """
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    # Define the condition for Vertex AI Pipeline Job creation
    condition = {
        "display_name": "Vertex AI Retraining Job Created",
        "condition_threshold": {
            "filter": 'resource.type="aiplatform.googleapis.com/PipelineJob" AND protoPayload.methodName="google.cloud.aiplatform.v1.PipelineService.CreatePipelineJob"',
            "comparison": monitoring_v3.ComparisonType.COMPARISON_GT,
            "threshold_value": 0,
            "duration": {"seconds": 60},  # Check every 60 seconds
        }
    }

    # Create the alert policy
    alert_policy = {
        "display_name": "Retraining Job Alert",
        "conditions": [condition],
        "notification_channels": [notification_channel],
        "enabled": True,
        "combiner": monitoring_v3.AlertPolicy.Combiner.OR
    }

    # Apply the policy
    policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )

    logger.info(f"Created retraining job alert policy: {policy.name}")

def monitor_and_trigger_retraining(project_id, model_name, accuracy_threshold, drift_threshold, gcs_input, pipeline_name, notification_channel):
    """
    Monitor model accuracy, data drift, and prediction drift, and trigger retraining when necessary.
    This will also set up alerts for retraining job creation.
    """
    # Check for accuracy degradation
    create_accuracy_degradation_alert(project_id, model_name, absolute_threshold=accuracy_threshold, degradation_rate_threshold=0.05)

    # Check for data drift and prediction drift
    data_drift_detected = detect_data_drift(project_id, model_name)
    prediction_drift_detected = detect_prediction_drift(project_id, model_name)

    if data_drift_detected or prediction_drift_detected:
        logger.warning(f"Drift detected for {model_name}. Triggering retraining pipeline.")
        
        # Trigger the retraining pipeline
        pipeline_job_id = trigger_retraining_pipeline(project_id, pipeline_name, gcs_input, model_name)
        
        # Set up retraining job alert to notify when the retraining job is created
        setup_retraining_job_alert(project_id, notification_channel)

        logger.info(f"Retraining job triggered: {pipeline_job_id}")
    else:
        logger.info(f"No drift detected for {model_name}. No retraining needed.")

    logger.info("Model performance and drift monitoring completed.")
    

if __name__ == '__main__':
    import argparse

    # Parse arguments for monitoring setup
    parser = argparse.ArgumentParser(description='Setup Vertex AI monitoring, drift detection, and rollback with retraining')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model_name', required=True, help='Vertex AI model name')
    parser.add_argument('--endpoint_name', required=True, help='Vertex AI endpoint name')
    parser.add_argument('--sampling_rate', type=float, default=1.0, help='Sampling rate for request/response logging')
    parser.add_argument('--absolute_threshold', type=float, default=0.85, help='Absolute accuracy threshold (e.g., 0.85)')
    parser.add_argument('--degradation_rate_threshold', type=float, default=0.05, help='Accuracy degradation rate threshold over time')
    parser.add_argument('--time_window', type=int, default=86400, help='Time window in seconds to monitor for degradation (default is 24 hours)')
    parser.add_argument('--drift_threshold', type=float, default=0.05, help='Data drift threshold for retraining')
    parser.add_argument('--gcs_input', required=True, help='GCS path to input data for retraining')
    parser.add_argument('--pipeline_name', required=True, help='Name of the Vertex AI pipeline for retraining')
    parser.add_argument('--notification_channel', required=True, help='Notification channel ID (for alerts)')
    parser.add_argument('--bucket_name', required=True, help='Cloud Storage bucket name')
    parser.add_argument('--schema_version', required=True, help='Schema version for validation')
    args = parser.parse_args()

    # Run Vertex AI monitoring functions
    setup_vertex_ai_monitoring(args.project_id, args.model_name)
    create_data_drift_alert(args.project_id, args.model_name)
    create_prediction_drift_alert(args.project_id, args.model_name)
    create_resource_utilization_alert(args.project_id, args.model_name)
    create_latency_alert(args.project_id, args.model_name)
    create_schema_drift_alert(args.project_id, args.model_name)
    create_accuracy_degradation_alert(args.project_id, args.model_name, args.absolute_threshold, args.degradation_rate_threshold, args.time_window)

    # Load baseline statistics and schema from GCS
    existing_stats, existing_schema = load_baseline_stats_and_schema(args.bucket_name, args.model_name, args.schema_version)

    # Compute and store current statistics
    current_stats, anomalies = compute_and_store_statistics(args.project_id, args.model_name, existing_stats, existing_schema)

    # Schema Drift Detection
    schema_drift_detected = detect_schema_drift(args.project_id, args.model_name, args.bucket_name, args.schema_version)
    if schema_drift_detected:
        print(f"Schema drift detected for model {args.model_name}. Logged to Google Cloud Monitoring.")

    # Detect data drift and prediction drift if baseline statistics exist
    if existing_stats:
        detect_data_drift(args.project_id, args.model_name, current_stats, existing_stats, args.drift_threshold)
        detect_prediction_drift(args.project_id, args.model_name, current_stats, existing_stats, args.drift_threshold)
    else:
        print("No existing statistics found. Current statistics will be used as the baseline for future comparisons.")

    # Run rollback monitoring after other checks
    print("Starting rollback monitoring...")
    monitor_and_log_rollbacks(args.project_id, args.endpoint_name)

    # Monitor model and trigger retraining if needed
    monitor_and_trigger_retraining(
        project_id=args.project_id,
        model_name=args.model_name,
        accuracy_threshold=args.absolute_threshold,
        drift_threshold=args.drift_threshold,
        gcs_input=args.gcs_input,
        pipeline_name=args.pipeline_name,
        notification_channel=args.notification_channel
    )

    print("Vertex AI monitoring, drift detection, rollback, and retraining setup completed successfully!")
