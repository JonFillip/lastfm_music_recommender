from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Artifact,
    Model
)
from typing import NamedTuple
from deployment.vertex_ai.vertex_ai_monitoring import (
    setup_vertex_ai_monitoring,
    create_data_drift_alert,
    create_prediction_drift_alert,
    create_resource_utilization_alert,
    create_latency_alert,
    create_accuracy_degradation_alert,
    create_schema_drift_alert,
    check_existing_statistics_and_schema,
    compute_and_store_statistics,
    detect_data_drift,
    detect_prediction_drift,
    detect_schema_drift,
)
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('kubeflow_monitoring')

OutputSpec = NamedTuple('OutputSpec', [
    ('data_drift_score', float),
    ('prediction_drift_score', float),
    ('schema_drift_detected', bool),
    ('accuracy_score', float),
    ('latency_ms', float),
])

@component(
    packages_to_install=[
        'google-cloud-monitoring',
        'google-cloud-storage',
        'google-cloud-bigquery',
        'google-cloud-aiplatform',
        'pandas',
        'scipy',
        'tensorflow-data-validation',
    ],
    base_image='python:3.10'
)
def monitor_model(
    project_id: str,
    model: Input[Model],
    model_name: str,
    endpoint_name: str,
    sampling_rate: float,
    schema_version: str,
    config: Input[Artifact],
    monitoring_output: Output[Artifact]
) -> OutputSpec:
    import json
    from google.cloud import aiplatform

    try:
        log_step(logger, "Setting up Vertex AI monitoring", "Model Monitoring")
        setup_vertex_ai_monitoring(project_id, model_name, endpoint_name)

        log_step(logger, "Creating monitoring alerts", "Model Monitoring")
        create_data_drift_alert(project_id, model_name)
        create_prediction_drift_alert(project_id, model_name)
        create_resource_utilization_alert(project_id, model_name)
        create_latency_alert(project_id, model_name)
        create_accuracy_degradation_alert(project_id, model_name)
        create_schema_drift_alert(project_id, model_name)

        log_step(logger, "Checking existing statistics and schema", "Model Monitoring")
        existing_stats, existing_schema = check_existing_statistics_and_schema(project_id, model_name)

        log_step(logger, "Computing and storing current statistics", "Model Monitoring")
        current_stats, anomalies = compute_and_store_statistics(project_id, model_name, existing_stats, existing_schema)

        data_drift_score = 0
        prediction_drift_score = 0
        schema_drift_detected = False
        accuracy_score = 0
        latency_ms = 0

        log_step(logger, "Detecting schema drift", "Model Monitoring")
        schema_drift_detected = detect_schema_drift(project_id, model_name, config.path, schema_version)

        if existing_stats:
            log_step(logger, "Detecting data drift", "Model Monitoring")
            data_drift_score = detect_data_drift(project_id, model_name, current_stats, existing_stats)
            
            log_step(logger, "Detecting prediction drift", "Model Monitoring")
            prediction_drift_score = detect_prediction_drift(project_id, model_name, current_stats, existing_stats)
        else:
            logger.info("No existing statistics found. Current statistics will be used as the baseline for future comparisons.")

        log_step(logger, "Evaluating model performance", "Model Monitoring")
        endpoint = aiplatform.Endpoint(endpoint_name)
        model_performance = endpoint.get_model_performance()
        accuracy_score = model_performance.get('accuracy', 0)
        latency_ms = model_performance.get('latency_ms', 0)

        monitoring_results = {
            "data_drift_score": data_drift_score,
            "prediction_drift_score": prediction_drift_score,
            "schema_drift_detected": schema_drift_detected,
            "accuracy_score": accuracy_score,
            "latency_ms": latency_ms,
            "anomalies": anomalies
        }

        with open(monitoring_output.path, 'w') as f:
            json.dump(monitoring_results, f, indent=2)

        logger.info("Vertex AI monitoring setup and checks completed successfully!")

        return OutputSpec(
            data_drift_score=data_drift_score,
            prediction_drift_score=prediction_drift_score,
            schema_drift_detected=schema_drift_detected,
            accuracy_score=accuracy_score,
            latency_ms=latency_ms
        )

    except Exception as e:
        log_error(logger, e, 'Model Monitoring')
        raise

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Monitor model component for Kubeflow')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model', required=True, help='Path to the model')
    parser.add_argument('--model_name', required=True, help='Vertex AI model name')
    parser.add_argument('--endpoint_name', required=True, help='Vertex AI endpoint name')
    parser.add_argument('--sampling_rate', type=float, default=1.0, help='Sampling rate for request/response logging')
    parser.add_argument('--schema_version', required=True, help='Version of the schema for validation')
    parser.add_argument('--config', required=True, help='Path to the config file')
    parser.add_argument('--monitoring_output', required=True, help='Path to save monitoring results')

    args = parser.parse_args()

    monitor_model(
        project_id=args.project_id,
        model=args.model,
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        sampling_rate=args.sampling_rate,
        schema_version=args.schema_version,
        config=args.config,
        monitoring_output=args.monitoring_output
    )
