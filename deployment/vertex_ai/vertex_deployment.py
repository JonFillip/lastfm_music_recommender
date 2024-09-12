import argparse
from typing import Tuple
from google.cloud import aiplatform
from google.cloud.devtools import cloudbuild_v1

from google.cloud import run_v2
from src.utils.logging_utils import setup_logger, log_error, log_step
from vertex_ai_monitoring import monitor_and_log_rollbacks, monitor_and_trigger_retraining

logger = setup_logger('vertex_ai_deployment')

def deploy_to_vertex_ai(project_id: str, model_path: str, endpoint_name: str, model_name: str, canary_traffic_percent: int = 10) -> Tuple[str, str]:
    """
    Deploy the model to Vertex AI using a canary deployment strategy, checking if an existing model is already deployed.
    If the new model fails, allow traffic rollback to the existing model.
    """
    try:
        log_step(logger, 'Model Deployment to Vertex AI', 'Serving')
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id)
        
        # Upload the model to Vertex AI
        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=model_path,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest"
        )
        
        # Retrieve the endpoint or create a new one if it doesn't exist
        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
        if not endpoints:
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
            traffic_split = {model.resource_name: 100}  # 100% traffic to the new model since no model exists
            logger.info("No existing models. Deploying new model with 100% traffic.")
        else:
            endpoint = endpoints[0]
            
            # Check if there is an existing model deployed
            if endpoint.traffic_split:
                current_model_id = list(endpoint.traffic_split.keys())[0]  # Assume single model in the endpoint
                previous_traffic_split = endpoint.traffic_split  # Save the current traffic split for rollback
                
                # Apply the canary strategy: split traffic between the existing model and the new model
                traffic_split = {
                    current_model_id: 100 - canary_traffic_percent,
                    model.resource_name: canary_traffic_percent
                }
                logger.info(f"Canary deployment: {100 - canary_traffic_percent}% to the current model, {canary_traffic_percent}% to the new model.")
            else:
                # No models currently deployed, assign 100% traffic to the new model
                traffic_split = {model.resource_name: 100}
                logger.info("No existing traffic split found. Assigning 100% traffic to the new model.")

        # Deploy the model to the endpoint with the traffic split
        model_deployment = model.deploy(
            endpoint=endpoint,
            machine_type="n1-standard-2",
            traffic_split=traffic_split,
            min_replica_count=1,
            max_replica_count=3,
            accelerator_type=None,
            accelerator_count=None,
            accelerator_config=None
        )

        logger.info(f"Model deployed to Vertex AI endpoint: {endpoint.resource_name}")

        # Use the consolidated function to monitor traffic split and set up rollback alerts
        monitor_and_log_rollbacks(project_id, endpoint_name)

        # After model deployment, monitor and trigger retraining if necessary
        monitor_and_trigger_retraining(
            project_id=project_id,
            model_name=model_name,
            accuracy_threshold=0.85,
            drift_threshold=0.05,
            gcs_input='gs://your_project/data/',
            pipeline_name='your_pipeline_name',
            notification_channel='your_notification_channel'
        )

        return (endpoint.resource_name, model.resource_name)
    
    except Exception as e:
        log_error(logger, e, 'Model Deployment to Vertex AI')
        
        # Rollback traffic to the previous model if deployment fails
        if endpoint and previous_traffic_split:
            endpoint.deploy(traffic_split=previous_traffic_split)
            logger.info("Rolled back traffic to the previous model due to deployment failure.")
        else:
            logger.error("No previous traffic split available for rollback.")
        raise


def setup_cloud_build_trigger(project_id: str, repo_name: str, branch_name: str, storage_bucket: str = None):
    """
    Set up a Cloud Build trigger for continuous training.
    The trigger can monitor both code changes in the repository and new data in a Cloud Storage bucket.
    
    :param project_id: GCP Project ID
    :param repo_name: Name of the GitHub repository
    :param branch_name: Branch to monitor for changes (e.g., 'main')
    :param storage_bucket: Optional. Name of the Cloud Storage bucket to monitor for new data (for retraining triggers).
    """
    client = cloudbuild_v1.CloudBuildClient()
    
    trigger = cloudbuild_v1.BuildTrigger()
    trigger.name = f"{repo_name}-{branch_name}-trigger"
    
    # Monitor for new commits or changes in the GitHub repository
    trigger.github = cloudbuild_v1.GitHubEventsConfig(
        owner="your-github-username",
        name=repo_name,
        push=cloudbuild_v1.PushFilter(
            branch=branch_name,
            included_paths=["pipeline/**", "model/**", "configs/**", "deployment/vertex_ai/**", "src/**", "data/**", "kubeflow/**"],  # Only triggers when changes are made in relevant directories
            ignored_paths=["README.md", "docs/**", "*.md"]  # Exempt README.md, docs, and markdown files changes from trigerring cloud build
        )
    )
    trigger.filename = "cloudbuild.yaml"
    
    # Optional: Monitor for new data ingestion in the Cloud Storage bucket
    if storage_bucket:
        # Add a notification filter for changes in the Cloud Storage bucket (retraining based on new data)
        trigger.gcs = cloudbuild_v1.StorageSource(bucket=storage_bucket)

    # Create the Cloud Build trigger
    operation = client.create_build_trigger(project_id=project_id, trigger=trigger)
    result = operation.result()
    
    print(f"Cloud Build trigger created: {result.name}")
    return result


def setup_cloud_run(project_id, service_name, image_url, region):
    client = run_v2.ServicesClient()
    
    service = run_v2.Service()
    service.template = run_v2.RevisionTemplate()
    service.template.containers = [
        run_v2.Container(
            image=image_url,
            env=[{"name": "ENV_VAR", "value": "production"}],  # Optional env vars
            resources=run_v2.ResourceRequirements(  # Optional resource settings
                limits={"cpu": "1", "memory": "512Mi"}
            )
        )
    ]
    
    parent = client.common_location_path(project_id, region)
    response = client.create_service(
        parent=parent,
        service=service,
        service_id=service_name
    )
    
    print(f"Cloud Run service created: {response.name}")
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy to Vertex AI and set up CI/CD')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model_path', required=True, help='Path to the model artifacts')
    parser.add_argument('--endpoint_name', required=True, help='Name for the Vertex AI endpoint')
    parser.add_argument('--repo_name', required=True, help='GitHub repository name')
    parser.add_argument('--branch_name', required=True, help='GitHub branch name')
    parser.add_argument('--service_name', required=True, help='Cloud Run service name')
    parser.add_argument('--image_url', required=True, help='Docker image URL for Cloud Run')
    parser.add_argument('--region', required=True, help='GCP region for deployment')
    parser.add_argument('--storage_bucket', required=True, help='Cloud Storage bucket to monitor for new data')
    
    args = parser.parse_args()
    
    # Deploy to Vertex AI
    endpoint = deploy_to_vertex_ai(args.project_id, args.model_path, args.endpoint_name)
    
    # Setup Cloud Build trigger
    trigger = setup_cloud_build_trigger(args.project_id, args.repo_name, args.branch_name, args.storage_bucket)
    
    # Setup Cloud Run service
    service = setup_cloud_run(args.project_id, args.service_name, args.image_url, args.region)
    
    print("Deployment completed successfully!")