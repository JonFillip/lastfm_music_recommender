import argparse
import time
import base64
import os
from typing import Tuple
from google.cloud import aiplatform, pubsub_v1
from google.cloud.devtools import cloudbuild_v1
from google.protobuf import duration_pb2
from google.cloud import run_v2
from src.utils.logging_utils import setup_logger, log_error, log_step
from vertex_ai_monitoring import monitor_and_log_rollbacks, monitor_and_trigger_retraining

logger = setup_logger('vertex_ai_deployment')

# Set your cooldown period (e.g., 5 minutes = 300 seconds)
COOLDOWN_PERIOD = 300  # Cooldown period in seconds

# Global variable to store the last build trigger timestamp
LAST_TRIGGER_TIME = 0

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
    Set up a Cloud Build trigger for continuous training with Pub/Sub and Cloud Function for cooldown.
    The trigger can monitor both code changes in the repository and new data in a Cloud Storage bucket.
    
    :param project_id: GCP Project ID
    :param repo_name: Name of the GitHub repository
    :param branch_name: Branch to monitor for changes (e.g., 'main')
    :param storage_bucket: Optional. Cloud Storage bucket to monitor for new data
    """
    client = cloudbuild_v1.CloudBuildClient()

    # Create the Cloud Build trigger
    trigger = cloudbuild_v1.BuildTrigger(
        name=f"{repo_name}-{branch_name}-trigger",
        github=cloudbuild_v1.GitHubEventsConfig(
            owner="your-github-username",
            name=repo_name,
            push=cloudbuild_v1.PushFilter(
                branch=branch_name,
                included_paths=["pipeline/**", "model/**", "configs/**", "deployment/vertex_ai/**", "src/**", "data/**", "kubeflow/**"],
                ignored_paths=["README.md", "docs/**", "*.md"]  # Exempt non-critical changes
            )
        ),
        filename="cloudbuild.yaml"
    )
    
    # Optional: Monitor for new data ingestion in the Cloud Storage bucket
    if storage_bucket:
        trigger.pubsub_config = cloudbuild_v1.PubsubConfig(
            topic=f"projects/{project_id}/topics/{storage_bucket}-trigger",
            subscription=f"projects/{project_id}/subscriptions/{storage_bucket}-trigger-sub"
        )

    # Create the Cloud Build trigger
    trigger_response = client.create_build_trigger(parent=f"projects/{project_id}", trigger=trigger)
    
    print(f"Cloud Build trigger created: {trigger_response.name}")

    # Set up build notifications
    notification_config = cloudbuild_v1.NotificationConfig(
        filter="build.status in (SUCCESS, FAILURE, INTERNAL_ERROR, TIMEOUT)",
        pubsub_topic=f"projects/{project_id}/topics/cloud-builds"
    )

    # Update the trigger with notification config
    trigger_response.notification_config = notification_config
    client.update_build_trigger(
        project_id=project_id,
        trigger_id=trigger_response.id,
        trigger=trigger_response
    )

    print("Build status notifications set up for successful and failed builds.")
    return trigger_response

def cloud_build_trigger(event, context):
    """
    Cloud Function to trigger a Cloud Build job with a cooldown period.
    It handles the Pub/Sub event and ensures that builds are not triggered
    more frequently than the specified cooldown period.
    """
    global LAST_TRIGGER_TIME
    current_time = time.time()

    # Decode the Pub/Sub message (if it's base64-encoded)
    if 'data' in event:
        data = base64.b64decode(event['data']).decode('utf-8')
        print(f"Received message: {data}")

    # Check if the cooldown period has passed
    if current_time - LAST_TRIGGER_TIME < COOLDOWN_PERIOD:
        print("Cooldown period not over. Skipping build trigger.")
        return

    # Trigger the Cloud Build job since the cooldown period has passed
    trigger_cloud_build()

    # Update the last trigger time
    LAST_TRIGGER_TIME = current_time

def trigger_cloud_build():
    """
    Function to trigger a Cloud Build job using the Cloud Build API.
    """
    client = cloudbuild_v1.CloudBuildClient()

    project_id = os.environ.get('PROJECT_ID')
    trigger_id = os.environ.get('TRIGGER_ID')
    model_name = os.environ.get('MODEL_NAME')
    endpoint_name = os.environ.get('ENDPOINT_NAME')

    # Trigger the Cloud Build job using the build trigger ID
    build = cloudbuild_v1.BuildTrigger(
        project_id=project_id,
        trigger_id=trigger_id
    )

    # Run the build
    client.run_build_trigger(project_id=project_id, trigger_id=trigger_id, source=None)
    print(f"Triggered Cloud Build job for trigger ID: {trigger_id}, Model: {model_name}, Endpoint: {endpoint_name}")


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
    import argparse
    from google.cloud import pubsub_v1
    import os

    # Parse arguments for Vertex AI and Cloud Build setup
    parser = argparse.ArgumentParser(description='Deploy to Vertex AI, set up Cloud Build triggers, and configure CI/CD with Cloud Run and Pub/Sub for continuous training')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model_name', required=True, help='Name of the machine learning model')  # Model name argument added
    parser.add_argument('--model_path', required=True, help='Path to the model artifacts')
    parser.add_argument('--endpoint_name', required=True, help='Name for the Vertex AI endpoint')
    parser.add_argument('--repo_name', required=True, help='GitHub repository name')
    parser.add_argument('--branch_name', required=True, help='GitHub branch name')
    parser.add_argument('--service_name', required=True, help='Cloud Run service name')
    parser.add_argument('--image_url', required=True, help='Docker image URL for Cloud Run')
    parser.add_argument('--region', required=True, help='GCP region for deployment')
    parser.add_argument('--storage_bucket', required=True, help='Cloud Storage bucket to monitor for new data')
    parser.add_argument('--trigger_id', required=True, help='Cloud Build trigger ID for retraining jobs')
    parser.add_argument('--cooldown_period', type=int, default=300, help='Cooldown period in seconds between Cloud Build jobs')
    parser.add_argument('--notification_channel', required=True, help='Notification channel ID for build status notifications')
    parser.add_argument('--canary_traffic_percent', type=int, default=10, help='Canary traffic split percentage')

    args = parser.parse_args()

    # Step 1: Deploy the model to Vertex AI
    print(f"Deploying model '{args.model_name}' to Vertex AI...")
    endpoint_name, model_name = deploy_to_vertex_ai(
        project_id=args.project_id,
        model_path=args.model_path,
        endpoint_name=args.endpoint_name,
        model_name=args.model_name,
        canary_traffic_percent=args.canary_traffic_percent
    )

    # Step 2: Set up Cloud Build trigger
    print("Setting up Cloud Build trigger for continuous training...")
    trigger_response = setup_cloud_build_trigger(
        project_id=args.project_id,
        repo_name=args.repo_name,
        branch_name=args.branch_name,
        storage_bucket=args.storage_bucket
    )

    # Step 3: Deploy the Cloud Function for cooldown (Pub/Sub)
    print("Setting up Pub/Sub topic and deploying Cloud Function for cooldown mechanism...")

    # Ensure the Pub/Sub topic exists
    pubsub_client = pubsub_v1.PublisherClient()
    topic_path = pubsub_client.topic_path(args.project_id, 'cloud-build-trigger')
    pubsub_client.create_topic(name=topic_path)

    # Deploy Cloud Function for cooldown
    os.system(f"gcloud functions deploy cloud_build_trigger --runtime python39 "
            f"--trigger-topic cloud-build-trigger "
            f"--set-env-vars PROJECT_ID={args.project_id},TRIGGER_ID={args.trigger_id} "
            f"--memory=128MB --timeout=300s")

    # Step 4: Set up Cloud Run service for deployment
    print("Setting up Cloud Run service for deployment...")
    service_response = setup_cloud_run(
        project_id=args.project_id,
        service_name=args.service_name,
        image_url=args.image_url,
        region=args.region
    )

    # Output results
    print(f"Deployment to Vertex AI completed. Endpoint: {endpoint_name}, Model: {model_name}")
    print(f"Cloud Build trigger '{trigger_response.name}' created.")
    print(f"Cloud Run service '{service_response.name}' created.")
    print("MLOps pipeline with Cloud Build, Pub/Sub, Cloud Function cooldown, and Cloud Run setup completed successfully.")
