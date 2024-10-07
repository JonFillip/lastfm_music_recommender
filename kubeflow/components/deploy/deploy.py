from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Model,
    Artifact,
)
from typing import NamedTuple
from deployment.vertex_ai.vertex_deployment import (
    deploy_to_vertex_ai,
    setup_cloud_build_trigger,
    setup_cloud_run,
    trigger_cloud_build
)
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('kubeflow_deploy_component')

# Define the OutputSpec NamedTuple
OutputSpec = NamedTuple('OutputSpec', [('endpoint', str), ('model', str), ('cloud_run_service', str)])

@component(
    packages_to_install=['google-cloud-aiplatform', 'google-cloud-build', 'google-cloud-run'],
    base_image='python:3.10'
)
def deploy_model(
    project_id: str,
    model_path: Input[Model],
    model_name: str,
    endpoint_name: str,
    repo_name: str,
    branch_name: str,
    service_name: str,
    image_url: str,
    region: str,
    storage_bucket: str,
    trigger_id: str,
    notification_channel: str,
    deployment_info: Output[Artifact],
    canary_traffic_percent: int = 10,
    cooldown_period: int = 300
) -> OutputSpec:
    """
    Kubeflow component to deploy a model to Vertex AI, set up CI/CD with Cloud Build and Cloud Run.
    """
    import json
    import os

    try:
        log_step(logger, "Deploying model to Vertex AI", "Model Deployment")
        endpoint, model = deploy_to_vertex_ai(
            project_id=project_id,
            model_path=model_path.uri,
            endpoint_name=endpoint_name,
            model_name=model_name,
            canary_traffic_percent=canary_traffic_percent
        )

        log_step(logger, "Setting up Cloud Build trigger", "CI/CD Setup")
        trigger_response = setup_cloud_build_trigger(
            project_id=project_id,
            repo_name=repo_name,
            branch_name=branch_name,
            storage_bucket=storage_bucket
        )

        log_step(logger, "Setting up Cloud Run service", "CI/CD Setup")
        service_response = setup_cloud_run(
            project_id=project_id,
            service_name=service_name,
            image_url=image_url,
            region=region
        )

        log_step(logger, "Setting up Cloud Function for cooldown", "CI/CD Setup")
        os.system(f"gcloud functions deploy cloud_build_trigger --runtime python39 "
                f"--trigger-topic cloud-build-trigger "
                f"--set-env-vars PROJECT_ID={project_id},TRIGGER_ID={trigger_id},"
                f"MODEL_NAME={model_name},ENDPOINT_NAME={endpoint_name} "
                f"--memory=128MB --timeout=300s")

        # Write deployment info to output
        deployment_info_dict = {
            "endpoint_name": endpoint,
            "model_name": model,
            "cloud_build_trigger": trigger_response.name,
            "cloud_run_service": service_response.name,
            "canary_traffic_percent": canary_traffic_percent
        }
        with open(deployment_info.path, 'w') as f:
            json.dump(deployment_info_dict, f)

        logger.info("Deployment and CI/CD setup completed successfully!")
        return OutputSpec(endpoint=endpoint, model=model, cloud_run_service=service_response.name)
    
    except Exception as e:
        log_error(logger, e, 'Model Deployment and CI/CD Setup')
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy model component for Kubeflow')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model_path', required=True, help='Path to the model artifacts')
    parser.add_argument('--model_name', required=True, help='Name for the deployed model')
    parser.add_argument('--endpoint_name', required=True, help='Name for the Vertex AI endpoint')
    parser.add_argument('--repo_name', required=True, help='GitHub repository name')
    parser.add_argument('--branch_name', required=True, help='GitHub branch name')
    parser.add_argument('--service_name', required=True, help='Cloud Run service name')
    parser.add_argument('--image_url', required=True, help='Docker image URL for Cloud Run')
    parser.add_argument('--region', required=True, help='GCP region for deployment')
    parser.add_argument('--storage_bucket', required=True, help='Cloud Storage bucket to monitor for new data')
    parser.add_argument('--trigger_id', required=True, help='Cloud Build trigger ID for retraining jobs')
    parser.add_argument('--notification_channel', required=True, help='Notification channel ID for build status notifications')
    parser.add_argument('--canary_traffic_percent', type=int, default=10, help='Percentage of traffic to route to the new model')
    parser.add_argument('--cooldown_period', type=int, default=300, help='Cooldown period in seconds between Cloud Build jobs')
    parser.add_argument('--deployment_info', required=True, help='Path to save deployment info')
    
    args = parser.parse_args()
    
    deploy_model(
        project_id=args.project_id,
        model_path=args.model_path,
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        repo_name=args.repo_name,
        branch_name=args.branch_name,
        service_name=args.service_name,
        image_url=args.image_url,
        region=args.region,
        storage_bucket=args.storage_bucket,
        trigger_id=args.trigger_id,
        notification_channel=args.notification_channel,
        canary_traffic_percent=args.canary_traffic_percent,
        cooldown_period=args.cooldown_period,
        deployment_info=args.deployment_info
    )
