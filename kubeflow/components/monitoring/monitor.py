from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Artifact,
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

OutputSpec = NamedTuple('OutputSpec', [
    ('data_drift_score', float),
    ('prediction_drift_score', float),
    ('schema_drift_detected', bool),  # Added schema drift detection result
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
    base_image='python:3.9'
)
def monitor_model(
    project_id: str,
    model_name: str,
    sampling_rate: float,
    schema_version: str,  # Added schema version
    config: Input[Artifact],
) -> OutputSpec:
    import json

    # Setup monitoring and alerts
    setup_vertex_ai_monitoring(project_id, model_name)
    create_data_drift_alert(project_id, model_name)
    create_prediction_drift_alert(project_id, model_name)
    create_resource_utilization_alert(project_id, model_name)
    create_latency_alert(project_id, model_name)
    create_schema_drift_alert(project_id, model_name)  # New schema drift alert

    # Check for existing statistics and schema
    existing_stats, existing_schema = check_existing_statistics_and_schema(project_id, model_name)

    # Compute and store current statistics
    current_stats, anomalies = compute_and_store_statistics(project_id, model_name, existing_stats, existing_schema)

    data_drift_score = 0
    prediction_drift_score = 0
    schema_drift_detected = False  # Variable to store schema drift detection result

    # Detect schema drift
    schema_drift_detected = detect_schema_drift(project_id, model_name, config.path, schema_version)

    # Detect data drift and prediction drift if baseline statistics exist
    if existing_stats:
        data_drift_score = detect_data_drift(project_id, model_name, current_stats, existing_stats)
        prediction_drift_score = detect_prediction_drift(project_id, model_name, current_stats, existing_stats)
    else:
        print("No existing statistics found. Current statistics will be used as the baseline for future comparisons.")

    print("Vertex AI monitoring setup and checks completed successfully!")

    return (data_drift_score, prediction_drift_score, schema_drift_detected)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Monitor model component for Kubeflow')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model_name', required=True, help='Vertex AI model name')
    parser.add_argument('--sampling_rate', type=float, default=1.0, help='Sampling rate for request/response logging')
    parser.add_argument('--schema_version', required=True, help='Version of the schema for validation')  # Added schema_version argument
    parser.add_argument('--config', required=True, help='Path to the config file')

    args = parser.parse_args()

    monitor_model(
        project_id=args.project_id,
        model_name=args.model_name,
        sampling_rate=args.sampling_rate,
        schema_version=args.schema_version,  # Pass schema version to the component
        config=args.config,
    )
