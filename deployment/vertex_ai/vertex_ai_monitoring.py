import tensorflow_data_validation as tfdv
import argparse
import json
import pandas as pd
import datetime
import random
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import ks_2samp
from google.cloud import monitoring_v3, storage, aiplatform
from google.api import metric_pb2 as ga_metric
from google.protobuf import duration_pb2
from src.data_processing.data_validation import (
    load_statistics_from_gcs,
    load_schema_from_gcs,
    compare_schemas,
    save_statistics_to_gcs
)
from src.utils.logging_utils import setup_logger, log_error, log_step
from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2

logger = setup_logger('vertex_ai_pipeline_monitoring')

class VertexAIMonitoring:
    def __init__(self, project_id: str, model_name: str, bucket_name: str,
                mlmd_host: str, mlmd_port: int, mlmd_database: str,
                mlmd_user: str, mlmd_password: str):
        self.project_id = project_id
        self.model_name = model_name
        self.bucket_name = bucket_name
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"
        self.feature_store_client = aiplatform.gapic.FeaturestoreServiceClient()

        # Connect to MLMD using PostgreSQL
        self.mlmd_connection_config = metadata_store_pb2.ConnectionConfig()
        self.mlmd_connection_config.postgresql.host = mlmd_host
        self.mlmd_connection_config.postgresql.port = mlmd_port
        self.mlmd_connection_config.postgresql.database = mlmd_database
        self.mlmd_connection_config.postgresql.user = mlmd_user
        self.mlmd_connection_config.postgresql.password = mlmd_password
        self.mlmd_store = metadata_store.MetadataStore(self.mlmd_connection_config)

    def setup_custom_metrics(self) -> None:
        """Sets up custom metrics for Vertex AI monitoring in Cloud Monitoring."""
        metrics = [
            self._create_metric_descriptor("prediction_drift", "Prediction drift metric"),
            self._create_metric_descriptor("data_drift", "Data drift metric"),
            self._create_metric_descriptor("prediction_latency", "Latency of prediction requests", value_type=ga_metric.MetricDescriptor.ValueType.INT64),
            self._create_metric_descriptor("accuracy", "Accuracy of the model"),
            self._create_metric_descriptor("schema_drift", "Schema drift metric", value_type=ga_metric.MetricDescriptor.ValueType.INT64),
            self._create_metric_descriptor("missing_schema", "Indicates missing baseline schema", value_type=ga_metric.MetricDescriptor.ValueType.INT64),
            self._create_metric_descriptor("missing_statistics", "Indicates missing baseline statistics", value_type=ga_metric.MetricDescriptor.ValueType.INT64),
            self._create_metric_descriptor("feature_store_read_count", "Number of read operations from the feature store", value_type=ga_metric.MetricDescriptor.ValueType.INT64),
            self._create_metric_descriptor("feature_store_write_count", "Number of write operations to the feature store", value_type=ga_metric.MetricDescriptor.ValueType.INT64),
            self._create_metric_descriptor("feature_store_latency", "Latency of feature store operations", value_type=ga_metric.MetricDescriptor.ValueType.DISTRIBUTION),
        ]

        for metric in metrics:
            try:
                descriptor = self.client.create_metric_descriptor(
                    name=self.project_name,
                    metric_descriptor=metric
                )
                logger.info(f"Created metric descriptor: {descriptor.name}")
            except Exception as e:
                logger.warning(f"Metric descriptor {metric.type} already exists or could not be created. Error: {e}")

    def _create_metric_descriptor(self, metric_name: str, description: str, value_type: int = ga_metric.MetricDescriptor.ValueType.DOUBLE) -> ga_metric.MetricDescriptor:
        return ga_metric.MetricDescriptor(
            type=f"custom.googleapis.com/vertex_ai/{self.model_name}/{metric_name}",
            metric_kind=ga_metric.MetricDescriptor.MetricKind.GAUGE,
            value_type=value_type,
            description=description
        )

    def create_alert_policy(self, display_name: str, filter_str: str, threshold: float, duration_seconds: int, comparison: int, notification_channel_id: str) -> None:
        """Creates an alert policy in Google Cloud Monitoring."""
        client = monitoring_v3.AlertPolicyServiceClient()
        condition = monitoring_v3.AlertPolicy.Condition(
            display_name=display_name,
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter=filter_str,
                comparison=comparison,
                threshold_value=threshold,
                duration=duration_pb2.Duration(seconds=duration_seconds),
            ),
        )

        alert_policy = monitoring_v3.AlertPolicy(
            display_name=f"{self.model_name} {display_name}",
            conditions=[condition],
            notification_channels=[notification_channel_id],
            combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
        )

        policy = client.create_alert_policy(
            name=self.project_name,
            alert_policy=alert_policy
        )
        logger.info(f"Created alert policy: {policy.name}")

    def create_accuracy_degradation_alert(self, absolute_threshold: float, degradation_rate_threshold: float, time_window_seconds: int, notification_channel_id: str) -> None:
        """Creates an alert for accuracy degradation."""
        self.create_alert_policy(
            "Accuracy below absolute threshold",
            f'metric.type="custom.googleapis.com/vertex_ai/{self.model_name}/accuracy"',
            absolute_threshold,
            300,
            monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_LT,
            notification_channel_id
        )
        self.create_alert_policy(
            "Accuracy degradation over time",
            f'metric.type="custom.googleapis.com/vertex_ai/{self.model_name}/accuracy"',
            degradation_rate_threshold,
            time_window_seconds,
            monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_LT,
            notification_channel_id
        )

    def create_resource_utilization_alert(self, notification_channel_id: str) -> None:
        """Creates alerts for resource utilization (CPU, memory, and GPU)."""
        resources = [
            ("CPU", "compute.googleapis.com/instance/cpu/utilization"),
            ("Memory", "compute.googleapis.com/instance/memory/utilization"),
            ("GPU", "compute.googleapis.com/instance/gpu/utilization")
        ]

        for resource_name, metric_type in resources:
            self.create_alert_policy(
                f"High {resource_name} utilization",
                f'metric.type="{metric_type}"',
                0.8,  # 80% utilization threshold
                300,  # 5 minutes duration
                monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_GT,
                notification_channel_id
            )

    def create_rollback_alert_policy(self, notification_channel_id: str) -> None:
        """Creates an alert policy for rollback detection based on traffic split anomalies."""
        filter_str = f'metric.type="custom.googleapis.com/vertex_ai/{self.model_name}/traffic_anomaly"'
        self.create_alert_policy(
            display_name="Rollback Detection Alert",
            filter_str=filter_str,
            threshold=0,  # Threshold set to detect any anomaly (since value will be 1 when anomaly is detected)
            duration_seconds=300,
            comparison=monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_GT,
            notification_channel_id=notification_channel_id
        )

    def log_metric(self, metric_name: str, value: Union[float, int]) -> None:
        """Logs a metric to Google Cloud Monitoring."""
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/vertex_ai/{self.model_name}/{metric_name}"
        series.resource.type = "global"
        series.resource.labels["project_id"] = self.project_id
        point = monitoring_v3.Point()
        if isinstance(value, float):
            point.value.double_value = value
        else:
            point.value.int64_value = value
        now = datetime.datetime.now()
        point.interval.end_time.seconds = int(now.timestamp())
        point.interval.end_time.nanos = int((now.timestamp() - int(now.timestamp())) * 10**9)
        series.points = [point]
        self.client.create_time_series(name=self.project_name, time_series=[series])
        logger.info(f"Logged {metric_name} for model {self.model_name}: {value}")

    def detect_data_drift(self, drift_threshold: float) -> Optional[float]:
        """Detects data drift by comparing current (serving) statistics with baseline (training) statistics."""
        try:
            log_step(logger, 'Detecting Data Drift', 'Data Drift Detection')
            today = datetime.datetime.now().strftime("%Y%m%d")

            baseline_stats = load_statistics_from_gcs(self.bucket_name, self.model_name, 'train', today)
            if not baseline_stats:
                self.log_metric("missing_statistics", 1)
                return None

            serving_stats = load_statistics_from_gcs(self.bucket_name, self.model_name, 'serving', today)
            if not serving_stats:
                self.log_metric("missing_statistics", 1)
                return None

            schema = load_schema_from_gcs(self.bucket_name, self.model_name, 'current')
            if not schema:
                self.log_metric("missing_schema", 1)
                return None

            # Compare the statistics
            anomalies = tfdv.validate_statistics(statistics=serving_stats, schema=schema, previous_statistics=baseline_stats)

            drift_score = len(anomalies.anomaly_info)
            for feature_name, anomaly_info in anomalies.anomaly_info.items():
                logger.warning(f"Data drift detected for feature {feature_name}: {anomaly_info.description}")
                self.log_metric("data_drift", 1)

            if drift_score > drift_threshold:
                logger.warning(f"Significant data drift detected. Drift score: {drift_score} > {drift_threshold}")
            else:
                logger.info(f"No significant data drift detected. Drift score: {drift_score} <= {drift_threshold}")

            # Log drift detection results to MLMD
            self._log_drift_detection_to_mlmd(drift_score, drift_threshold)

            return drift_score

        except Exception as e:
            log_error(logger, e, 'Data Drift Detection')
            return None

    def _log_drift_detection_to_mlmd(self, drift_score: float, drift_threshold: float):
        """Log drift detection results to ML Metadata."""
        execution = metadata_store_pb2.Execution()
        execution.type_id = self._get_or_create_execution_type_id("DataDriftDetection")
        execution.properties["model_name"].string_value = self.model_name
        execution.properties["drift_score"].double_value = drift_score
        execution.properties["drift_threshold"].double_value = drift_threshold
        execution.properties["timestamp"].string_value = datetime.datetime.now().isoformat()

        execution_id = self.mlmd_store.put_executions([execution])[0]
        logger.info(f"Logged drift detection results to MLMD with execution ID: {execution_id}")

    def _get_or_create_execution_type_id(self, type_name: str) -> int:
        """Helper method to get or create an execution type ID."""
        try:
            execution_type = self.mlmd_store.get_execution_type(type_name)
        except metadata_store.errors.NotFoundError:
            execution_type = metadata_store_pb2.ExecutionType(name=type_name)
            self.mlmd_store.put_execution_type(execution_type)
        return execution_type.id

    def detect_prediction_drift(self, drift_threshold: float) -> Optional[float]:
        """Detects prediction drift using the Kolmogorov-Smirnov (KS) test."""
        try:
            log_step(logger, 'Detecting Prediction Drift', 'Prediction Drift Detection')
            today = datetime.datetime.now().strftime("%Y%m%d")

            train_stats = load_statistics_from_gcs(self.bucket_name, self.model_name, 'train', today)
            if not train_stats:
                self.log_metric("missing_statistics", 1)
                return None

            serving_stats = load_statistics_from_gcs(self.bucket_name, self.model_name, 'serving', today)
            if not serving_stats:
                self.log_metric("missing_statistics", 1)
                return None

            # Assuming 'prediction' is the feature containing model predictions
            train_predictions = tfdv.get_feature_stats_as_dataframe(train_stats)
            serving_predictions = tfdv.get_feature_stats_as_dataframe(serving_stats)

            if 'prediction' not in train_predictions.columns or 'prediction' not in serving_predictions.columns:
                logger.error("Prediction feature not found in statistics.")
                return None

            statistic, _ = ks_2samp(train_predictions['prediction'], serving_predictions['prediction'])

            self.log_metric("prediction_drift", statistic)

            if statistic > drift_threshold:
                logger.warning(f"Prediction drift detected: KS statistic = {statistic}")
            else:
                logger.info(f"No significant prediction drift detected")

            # Log prediction drift results to MLMD
            self._log_prediction_drift_to_mlmd(statistic, drift_threshold)

            return statistic

        except Exception as e:
            log_error(logger, e, 'Prediction Drift Detection')
            return None

    def _log_prediction_drift_to_mlmd(self, statistic: float, drift_threshold: float):
        """Log prediction drift results to ML Metadata."""
        execution = metadata_store_pb2.Execution()
        execution.type_id = self._get_or_create_execution_type_id("PredictionDriftDetection")
        execution.properties["model_name"].string_value = self.model_name
        execution.properties["ks_statistic"].double_value = statistic
        execution.properties["drift_threshold"].double_value = drift_threshold
        execution.properties["timestamp"].string_value = datetime.datetime.now().isoformat()

        execution_id = self.mlmd_store.put_executions([execution])[0]
        logger.info(f"Logged prediction drift results to MLMD with execution ID: {execution_id}")

    def detect_schema_drift(self, schema_version: str) -> Optional[bool]:
        """Detects schema drift by comparing the current schema with the baseline (training) schema."""
        try:
            log_step(logger, 'Detecting Schema Drift', 'Schema Drift Detection')

            baseline_schema = load_schema_from_gcs(self.bucket_name, self.model_name, schema_version)
            if not baseline_schema:
                self.log_metric("missing_schema", 1)
                return None

            current_schema = load_schema_from_gcs(self.bucket_name, self.model_name, 'current')
            if not current_schema:
                self.log_metric("missing_schema", 1)
                return None

            schema_drift_detected = compare_schemas(baseline_schema, current_schema)

            if schema_drift_detected:
                logger.info(f"Schema drift detected for model {self.model_name}.")
                self.log_metric("schema_drift", 1)
            else:
                logger.info(f"No schema drift detected for model {self.model_name}.")
                self.log_metric("schema_drift", 0)

            # Log schema drift results to MLMD
            self._log_schema_drift_to_mlmd(schema_drift_detected)

            return schema_drift_detected

        except Exception as e:
            log_error(logger, e, 'Schema Drift Detection')
            return None

    def _log_schema_drift_to_mlmd(self, schema_drift_detected: bool):
        """Log schema drift results to ML Metadata."""
        execution = metadata_store_pb2.Execution()
        execution.type_id = self._get_or_create_execution_type_id("SchemaDriftDetection")
        execution.properties["model_name"].string_value = self.model_name
        execution.properties["schema_drift_detected"].int_value = int(schema_drift_detected)
        execution.properties["timestamp"].string_value = datetime.datetime.now().isoformat()

        execution_id = self.mlmd_store.put_executions([execution])[0]
        logger.info(f"Logged schema drift results to MLMD with execution ID: {execution_id}")

    def monitor_traffic_split(self, endpoint_name: str, expected_traffic_split: Dict[str, int]) -> Optional[Dict[str, int]]:
        """Monitor the traffic split in Vertex AI to detect rollback."""
        try:
            log_step(logger, 'Monitoring traffic split', 'Rollback Monitoring')

            aiplatform.init(project=self.project_id)

            endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
            if not endpoints:
                log_error(logger, f"Endpoint {endpoint_name} not found.", "Rollback Monitoring")
                return None
            endpoint = endpoints[0]

            traffic_split = endpoint.gca_resource.traffic_split
            total_traffic = sum(traffic_split.values())
            anomaly_detected = False

            for model_id, traffic_percentage in traffic_split.items():
                logger.info(f"Model {model_id} is receiving {traffic_percentage}% of the traffic.")

                # Check against expected traffic split
                expected_percentage = expected_traffic_split.get(model_id, 0)
                if traffic_percentage != expected_percentage:
                    anomaly_detected = True
                    logger.warning(f"Anomaly detected: Model {model_id} traffic is {traffic_percentage}%, expected {expected_percentage}%.")

            if total_traffic != 100:
                anomaly_detected = True
                logger.warning("Traffic split does not sum to 100%, indicating a possible rollback or misconfiguration.")

            # Log traffic anomaly metric
            self.log_metric("traffic_anomaly", int(anomaly_detected))

            # Log traffic split to MLMD
            self._log_traffic_split_to_mlmd(traffic_split, anomaly_detected)

            return traffic_split
        except Exception as e:
            log_error(logger, e, "Rollback Monitoring")
            return None

    def _log_traffic_split_to_mlmd(self, traffic_split: Dict[str, int], anomaly_detected: bool):
        """Log traffic split to ML Metadata."""
        execution = metadata_store_pb2.Execution()
        execution.type_id = self._get_or_create_execution_type_id("TrafficSplitMonitoring")
        execution.properties["model_name"].string_value = self.model_name
        execution.properties["anomaly_detected"].int_value = int(anomaly_detected)
        for model_id, percentage in traffic_split.items():
            execution.custom_properties[f"traffic_{model_id}"].double_value = percentage
        execution.properties["timestamp"].string_value = datetime.datetime.utcnow().isoformat() + 'Z'

        execution_id = self.mlmd_store.put_executions([execution])[0]
        logger.info(f"Logged traffic split to MLMD with execution ID: {execution_id}")

    def trigger_retraining_pipeline(self, pipeline_name: str, gcs_input: str) -> str:
        """Trigger a Vertex AI pipeline for continuous retraining."""
        aiplatform.init(project=self.project_id)

        pipeline_params = {
            'input_data': gcs_input,
            'model_name': self.model_name
        }

        pipeline_job = aiplatform.PipelineJob(
            display_name=f'Retraining - {self.model_name}',
            template_path=pipeline_name,
            parameter_values=pipeline_params
        )

        pipeline_job.run()
        logger.info(f"Triggered retraining pipeline for {self.model_name}.")

        # Log retraining trigger to MLMD
        self._log_retraining_trigger_to_mlmd(pipeline_job.resource_name)

        return pipeline_job.resource_name

    def _log_retraining_trigger_to_mlmd(self, pipeline_job_id: str):
        """Log retraining trigger to ML Metadata."""
        execution = metadata_store_pb2.Execution()
        execution.type_id = self._get_or_create_execution_type_id("RetrainingTrigger")
        execution.properties["model_name"].string_value = self.model_name
        execution.properties["pipeline_job_id"].string_value = pipeline_job_id
        execution.properties["timestamp"].string_value = datetime.datetime.now().isoformat()

        execution_id = self.mlmd_store.put_executions([execution])[0]
        logger.info(f"Logged retraining trigger to MLMD with execution ID: {execution_id}")

    def setup_retraining_job_alert(self, notification_channel_id: str) -> None:
        """Set up a Cloud Monitoring alert for Vertex AI retraining jobs."""
        client = monitoring_v3.AlertPolicyServiceClient()
        condition = monitoring_v3.AlertPolicy.Condition(
            display_name="Vertex AI Retraining Job Created",
            condition_monitoring_query_language=monitoring_v3.AlertPolicy.Condition.MonitoringQueryLanguageCondition(
                query=(
                    'fetch aiplatform.googleapis.com/pipeline_job '
                    '| {metric.type="aiplatform.googleapis.com/pipeline_job/pipeline_job_state"}'
                ),
                duration=duration_pb2.Duration(seconds=60),
                trigger=monitoring_v3.AlertPolicy.Condition.Trigger(count=1)
            )
        )

        alert_policy = monitoring_v3.AlertPolicy(
            display_name="Retraining Job Alert",
            conditions=[condition],
            notification_channels=[notification_channel_id],
            combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
            enabled=True
        )

        policy = client.create_alert_policy(
            name=self.project_name,
            alert_policy=alert_policy
        )

        logger.info(f"Created retraining job alert policy: {policy.name}")

    def monitor_and_trigger_retraining(self, accuracy_threshold: float, degradation_rate_threshold: float, drift_threshold: float, gcs_input: str, pipeline_name: str, notification_channel_id: str, time_window_seconds: int) -> None:
        """Monitor model accuracy, data drift, and prediction drift, and trigger retraining when necessary."""
        self.create_accuracy_degradation_alert(accuracy_threshold, degradation_rate_threshold, time_window_seconds, notification_channel_id)

        data_drift_score = self.detect_data_drift(drift_threshold)
        prediction_drift_statistic = self.detect_prediction_drift(drift_threshold)

        if (data_drift_score and data_drift_score > drift_threshold) or (prediction_drift_statistic and prediction_drift_statistic > drift_threshold):
            logger.warning(f"Drift detected for {self.model_name}. Triggering retraining pipeline.")

            pipeline_job_id = self.trigger_retraining_pipeline(pipeline_name, gcs_input)

            self.setup_retraining_job_alert(notification_channel_id)

            logger.info(f"Retraining job triggered: {pipeline_job_id}")
        else:
            logger.info(f"No drift detected for {self.model_name}. No retraining needed.")

        logger.info("Model performance and drift monitoring completed.")

    def log_feature_store_metric(self, feature_store_id: str, entity_type_id: str, metric_name: str, value: Union[int, float]):
        """Logs a feature store metric to Google Cloud Monitoring."""
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/vertex_ai/{self.model_name}/{metric_name}"
        series.resource.type = "aiplatform.googleapis.com/Featurestore"
        series.resource.labels["featurestore_id"] = feature_store_id
        series.resource.labels["entity_type_id"] = entity_type_id
        point = monitoring_v3.Point()
        if isinstance(value, int):
            point.value.int64_value = value
        else:
            point.value.double_value = value
        now = datetime.datetime.now()
        point.interval.end_time.seconds = int(now.timestamp())
        point.interval.end_time.nanos = int((now.timestamp() - int(now.timestamp())) * 10**9)
        series.points = [point]
        self.client.create_time_series(name=self.project_name, time_series=[series])
        logger.info(f"Logged feature store metric {metric_name} with value {value}")

    def monitor_feature_store(self, feature_store_id: str, entity_type_id: str):
        """Monitors the Feature Store and logs relevant metrics."""
        try:
            log_step(logger, 'Monitoring Feature Store', 'Feature Store Monitoring')

            featurestore_name = f"projects/{self.project_id}/locations/-/featurestores/{feature_store_id}"
            entity_type_name = f"{featurestore_name}/entityTypes/{entity_type_id}"

            # Log read and write counts
            # Placeholder implementation; actual read/write counts need to be retrieved from monitoring metrics or logs
            read_count = 100  # Replace with actual logic to get read count
            write_count = 50  # Replace with actual logic to get write count

            self.log_feature_store_metric(feature_store_id, entity_type_id, "feature_store_read_count", read_count)
            self.log_feature_store_metric(feature_store_id, entity_type_id, "feature_store_write_count", write_count)

            # Log latency (this is a placeholder, actual implementation may vary based on available metrics)
            avg_latency = 200  # Replace with actual logic to get average latency
            self.log_feature_store_metric(feature_store_id, entity_type_id, "feature_store_latency", avg_latency)

            logger.info(f"Monitored feature store {feature_store_id}, entity type {entity_type_id}")
        except Exception as e:
            log_error(logger, e, 'Feature Store Monitoring')

    def create_feature_store_alerts(self, feature_store_id: str, entity_type_id: str, notification_channel_id: str):
        """Creates alerts for Feature Store monitoring."""
        self.create_alert_policy(
            "High Feature Store Read Count",
            f'metric.type="custom.googleapis.com/vertex_ai/{self.model_name}/feature_store_read_count" AND resource.labels.featurestore_id="{feature_store_id}" AND resource.labels.entity_type_id="{entity_type_id}"',
            1000,  # Threshold: 1000 reads
            300,   # Duration: 5 minutes
            monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_GT,
            notification_channel_id
        )
        self.create_alert_policy(
            "High Feature Store Write Count",
            f'metric.type="custom.googleapis.com/vertex_ai/{self.model_name}/feature_store_write_count" AND resource.labels.featurestore_id="{feature_store_id}" AND resource.labels.entity_type_id="{entity_type_id}"',
            500,   # Threshold: 500 writes
            300,   # Duration: 5 minutes
            monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_GT,
            notification_channel_id
        )
        self.create_alert_policy(
            "High Feature Store Latency",
            f'metric.type="custom.googleapis.com/vertex_ai/{self.model_name}/feature_store_latency" AND resource.labels.featurestore_id="{feature_store_id}" AND resource.labels.entity_type_id="{entity_type_id}"',
            1000,  # Threshold: 1000 ms
            300,   # Duration: 5 minutes
            monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_GT,
            notification_channel_id
        )

def log_request_response(project_id: str, model_name: str, request: Dict, response: Dict, latency_ms: float, sampling_rate: float = 0.1) -> None:
    """Logs serving request/response data and latency to Cloud Storage with optional sampling."""
    if sampling_rate >= 1 or random.random() < sampling_rate:
        client = storage.Client(project=project_id)
        bucket_name = f"{project_id}-vertex-ai-logs"
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(f"{model_name}/logs/{datetime.datetime.now().isoformat()}.json")
        log_entry = {
            "request": request,
            "response": response,
            "latency_ms": latency_ms,
            "timestamp": datetime.datetime.now().isoformat()
        }
        blob.upload_from_string(json.dumps(log_entry))
        logger.info(f"Logged request/response for {model_name} (latency: {latency_ms}ms)")

def check_existing_statistics_and_schema(project_id: str, model_name: str, bucket_name: str, schema_version: str) -> Tuple[Optional[tfdv.types.DatasetFeatureStatisticsList], Optional[tfdv.types.Schema]]:
    today = datetime.datetime.now().strftime("%Y%m%d")

    try:
        existing_stats = load_statistics_from_gcs(bucket_name, model_name, 'serving', today)
    except Exception as e:
        logger.error(f"Error loading existing statistics: {e}")
        existing_stats = None

    try:
        schema = load_schema_from_gcs(bucket_name, model_name, schema_version)
    except Exception as e:
        logger.error(f"Error loading schema: {e}")
        schema = None

    return existing_stats, schema

def compute_and_store_statistics(project_id: str, model_name: str, bucket_name: str, existing_schema: Optional[tfdv.types.Schema]) -> Tuple[tfdv.types.DatasetFeatureStatisticsList, Optional[tfdv.types.Anomalies]]:
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(f"{project_id}-vertex-ai-logs")
    blobs = client.list_blobs(bucket_or_name=bucket, prefix=f"{model_name}/logs/")

    data = []
    for blob in blobs:
        content = json.loads(blob.download_as_string())
        data.append(content)

    df = pd.json_normalize(data)

    stats = tfdv.generate_statistics_from_dataframe(df)
    save_statistics_to_gcs(stats, bucket_name, model_name, 'serving')

    if existing_schema:
        anomalies = tfdv.validate_statistics(stats, schema=existing_schema)
    else:
        anomalies = None
        logger.warning("No existing schema found. Skipping anomaly detection.")

    return stats, anomalies

def main():
    parser = argparse.ArgumentParser(description='Setup Vertex AI monitoring, drift detection, and rollback with retraining')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model_name', required=True, help='Vertex AI model name')
    parser.add_argument('--endpoint_name', required=True, help='Vertex AI endpoint name')
    parser.add_argument('--absolute_threshold', type=float, default=0.85, help='Absolute accuracy threshold (e.g., 0.85)')
    parser.add_argument('--degradation_rate_threshold', type=float, default=0.05, help='Accuracy degradation rate threshold over time')
    parser.add_argument('--time_window', type=int, default=86400, help='Time window in seconds to monitor for degradation (default is 24 hours)')
    parser.add_argument('--drift_threshold', type=float, default=0.05, help='Data drift threshold for retraining')
    parser.add_argument('--gcs_input', required=True, help='GCS path to input data for retraining')
    parser.add_argument('--pipeline_name', required=True, help='Path to the Vertex AI pipeline template for retraining')
    parser.add_argument('--notification_channel', required=True, help='Notification channel ID (for alerts)')
    parser.add_argument('--bucket_name', required=True, help='Cloud Storage bucket name')
    parser.add_argument('--schema_version', required=True, help='Schema version for validation')
    parser.add_argument('--sampling_rate', type=float, default=0.1, help='Sampling rate for request/response logging')
    parser.add_argument('--feature_store_id', required=True, help='Vertex AI Feature Store ID')
    parser.add_argument('--entity_type_id', required=True, help='Entity Type ID in the Feature Store')
    parser.add_argument('--mlmd_host', required=True, help='MLMD PostgreSQL host')
    parser.add_argument('--mlmd_port', type=int, default=5432, help='MLMD PostgreSQL port')
    parser.add_argument('--mlmd_database', required=True, help='MLMD PostgreSQL database name')
    parser.add_argument('--mlmd_user', required=True, help='MLMD PostgreSQL username')
    parser.add_argument('--mlmd_password', required=True, help='MLMD PostgreSQL password')
    args = parser.parse_args()

    monitor = VertexAIMonitoring(
        project_id=args.project_id,
        model_name=args.model_name,
        bucket_name=args.bucket_name,
        mlmd_host=args.mlmd_host,
        mlmd_port=args.mlmd_port,
        mlmd_database=args.mlmd_database,
        mlmd_user=args.mlmd_user,
        mlmd_password=args.mlmd_password
    )
    
    monitor.setup_custom_metrics()
    monitor.create_alert_policy("Data Drift Alert",
                                f'metric.type="custom.googleapis.com/vertex_ai/{args.model_name}/data_drift"',
                                0.1, 300,
                                monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_GT,
                                args.notification_channel)
    monitor.create_alert_policy("Prediction Drift Alert",
                                f'metric.type="custom.googleapis.com/vertex_ai/{args.model_name}/prediction_drift"',
                                0.1, 300,
                                monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_GT,
                                args.notification_channel)
    monitor.create_resource_utilization_alert(args.notification_channel)
    monitor.create_alert_policy("Prediction Latency Alert",
                                f'metric.type="custom.googleapis.com/vertex_ai/{args.model_name}/prediction_latency"',
                                1000, 60,
                                monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_GT,
                                args.notification_channel)
    monitor.create_alert_policy("Schema Drift Alert",
                                f'metric.type="custom.googleapis.com/vertex_ai/{args.model_name}/schema_drift"',
                                1, 300,
                                monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_GT,
                                args.notification_channel)
    monitor.create_accuracy_degradation_alert(args.absolute_threshold, args.degradation_rate_threshold, args.time_window, args.notification_channel)
    monitor.create_feature_store_alerts(args.feature_store_id, args.entity_type_id, args.notification_channel)
    monitor.monitor_feature_store(args.feature_store_id, args.entity_type_id)
    monitor.create_rollback_alert_policy(args.notification_channel)

    existing_stats, existing_schema = check_existing_statistics_and_schema(args.project_id, args.model_name, args.bucket_name, args.schema_version)
    current_stats, anomalies = compute_and_store_statistics(args.project_id, args.model_name, args.bucket_name, existing_schema)

    monitor.detect_schema_drift(args.schema_version)
    monitor.detect_data_drift(args.drift_threshold)
    monitor.detect_prediction_drift(args.drift_threshold)

    monitor.monitor_traffic_split(args.endpoint_name)

    monitor.monitor_and_trigger_retraining(
        accuracy_threshold=args.absolute_threshold,
        degradation_rate_threshold=args.degradation_rate_threshold,
        drift_threshold=args.drift_threshold,
        gcs_input=args.gcs_input,
        pipeline_name=args.pipeline_name,
        notification_channel_id=args.notification_channel,
        time_window_seconds=args.time_window
    )

    logger.info("Vertex AI monitoring, drift detection, rollback, and retraining setup completed successfully!")

if __name__ == '__main__':
    main()
