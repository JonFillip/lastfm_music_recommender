import argparse
import time
import base64
import os
from typing import Tuple
from google.cloud import aiplatform, pubsub_v1, cloudbuild_v1, functions_v1, run_v2, firestore
from google.protobuf import field_mask_pb2
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('vertex_ai_deployment')

def deploy_to_vertex_ai(project_id: str, model_path: str, endpoint_name: str, model_name: str, canary_traffic_percent: int = 10) -> Tuple[str, str]:
    """
    Deploy the model to Vertex AI using a canary deployment strategy.
    If the new model fails, allow traffic rollback to the existing model.
    """
    try:
        log_step(logger, 'Model Deployment to Vertex AI', 'Serving')

        # Initialize Vertex AI
        aiplatform.init(project=project_id)

        # Upload the model to Vertex AI
        logger.debug(f"Uploading model '{model_name}' from path '{model_path}'")
        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=model_path,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest"
        )
        model.wait()
        logger.info(f"Model '{model_name}' uploaded successfully.")

        # Retrieve the endpoint or create a new one if it doesn't exist
        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
        if not endpoints:
            logger.info(f"Creating new endpoint '{endpoint_name}'")
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
            endpoint.wait()
            traffic_split = {model.resource_name: 100}  # 100% traffic to the new model
            logger.info("No existing models. Deploying new model with 100% traffic.")
        else:
            endpoint = endpoints[0]
            logger.info(f"Using existing endpoint '{endpoint.display_name}'")

            # Check if there is an existing model deployed
            if endpoint.traffic_split:
                current_model_id = list(endpoint.traffic_split.keys())[0]  # Assume single model in the endpoint
                previous_traffic_split = endpoint.traffic_split.copy()  # Save the current traffic split for rollback

                # Apply the canary strategy
                traffic_split = {
                    current_model_id: 100 - canary_traffic_percent,
                    model.resource_name: canary_traffic_percent
                }
                logger.info(f"Canary deployment: {100 - canary_traffic_percent}% to the current model, {canary_traffic_percent}% to the new model.")
            else:
                # No models currently deployed
                traffic_split = {model.resource_name: 100}
                logger.info("No existing traffic split found. Assigning 100% traffic to the new model.")

        # Deploy the model to the endpoint with the traffic split
        logger.debug("Deploying model to endpoint with specified traffic split.")
        model.deploy(
            endpoint=endpoint,
            machine_type="n1-standard-2",
            traffic_split=traffic_split,
            min_replica_count=1,
            max_replica_count=3
        )
        logger.info(f"Model deployed to Vertex AI endpoint: {endpoint.resource_name}")

        return (endpoint.resource_name, model.resource_name)

    except Exception as e:
        log_error(logger, e, 'Model Deployment to Vertex AI')

        # Rollback traffic to the previous model if deployment fails
        if 'endpoint' in locals() and 'previous_traffic_split' in locals():
            endpoint.update_traffic_split(traffic_split=previous_traffic_split)
            logger.info("Rolled back traffic to the previous model due to deployment failure.")
        else:
            logger.error("No previous traffic split available for rollback.")
        raise

def setup_cloud_build_trigger(project_id: str, repo_name: str, branch_name: str, storage_bucket: str = None):
    """
    Set up a Cloud Build trigger for continuous training.
    """
    try:
        log_step(logger, 'Setting up Cloud Build Trigger', 'CI/CD Pipeline')

        client = cloudbuild_v1.CloudBuildClient()

        trigger = cloudbuild_v1.BuildTrigger(
            name=f"{repo_name}-{branch_name}-trigger",
            github=cloudbuild_v1.GitHubEventsConfig(
                owner="your-github-username",
                name=repo_name,
                push=cloudbuild_v1.PushFilter(
                    branch=f'^{branch_name}$'  # Use regex for exact match
                )
            ),
            filename="cloudbuild.yaml",
            included_files=["pipeline/**", "model/**", "configs/**", "deployment/vertex_ai/**", "src/**", "data/**", "kubeflow/**"],
            ignored_files=["README.md", "docs/**", "*.md"]  # Exempt non-critical changes
        )

        # Optional: Monitor for new data ingestion in the Cloud Storage bucket
        if storage_bucket:
            trigger.pubsub_config = cloudbuild_v1.PubsubConfig(
                topic=f"projects/{project_id}/topics/{storage_bucket}-trigger"
            )

        # Create the Cloud Build trigger
        parent = f"projects/{project_id}/locations/global"
        trigger_response = client.create_build_trigger(parent=parent, trigger=trigger)
        logger.info(f"Cloud Build trigger created: {trigger_response.name}")

        # Set up build notifications
        notification_config = cloudbuild_v1.NotificationConfig(
            filter="build.status in (SUCCESS, FAILURE, INTERNAL_ERROR, TIMEOUT)",
            pubsub_topic=f"projects/{project_id}/topics/cloud-builds"
        )

        # Update the trigger with notification config
        trigger_response.trigger.notification_config = notification_config
        update_mask = field_mask_pb2.FieldMask(paths=['notification_config'])
        client.update_build_trigger(
            trigger=trigger_response.trigger,
            update_mask=update_mask
        )
        logger.info("Build status notifications set up for successful and failed builds.")

        return trigger_response.trigger

    except Exception as e:
        log_error(logger, e, 'Setting up Cloud Build Trigger')
        raise

def trigger_cloud_build():
    """
    Function to trigger a Cloud Build job using the Cloud Build API.
    """
    try:
        log_step(logger, 'Triggering Cloud Build Job', 'CI/CD Pipeline')

        client = cloudbuild_v1.CloudBuildClient()

        project_id = os.environ.get('PROJECT_ID')
        trigger_id = os.environ.get('TRIGGER_ID')
        model_name = os.environ.get('MODEL_NAME', 'default_model_name')
        endpoint_name = os.environ.get('ENDPOINT_NAME', 'default_endpoint_name')

        if not project_id or not trigger_id:
            raise ValueError("Environment variables 'PROJECT_ID' and 'TRIGGER_ID' must be set.")

        # Trigger the Cloud Build job
        operation = client.run_build_trigger(
            project_id=project_id,
            trigger_id=trigger_id,
            source=cloudbuild_v1.RepoSource()
        )

        build = operation.result()
        logger.info(f"Triggered Cloud Build job for trigger ID: {trigger_id}, Model: {model_name}, Endpoint: {endpoint_name}")
        logger.info(f"Build Status: {build.status.name}")

    except Exception as e:
        log_error(logger, e, 'Triggering Cloud Build Job')
        raise


def cloud_build_trigger(event, context):
    """
    Cloud Function to trigger a Cloud Build job with a cooldown period.
    It handles the Pub/Sub event and ensures that builds are not triggered
    more frequently than the specified cooldown period.
    """
    try:
        log_step(logger, 'Cloud Build Trigger Function Invoked', 'Cloud Function')

        # Initialize Firestore client
        firestore_client = firestore.Client()
        cooldown_collection = firestore_client.collection('cloud_build_cooldown')
        cooldown_doc = cooldown_collection.document('last_trigger_time')

        current_time = time.time()
        cooldown_period = int(os.environ.get('COOLDOWN_PERIOD', 300))  # Default to 300 seconds if not set

        @firestore.transactional
        def update_last_trigger_time(transaction):
            doc = cooldown_doc.get(transaction=transaction)
            if doc.exists:
                last_trigger_time = doc.to_dict().get('timestamp')
                if (current_time - last_trigger_time) < cooldown_period:
                    logger.info("Cooldown period not over. Skipping build trigger.")
                    return False
            # Update the last trigger time
            transaction.set(cooldown_doc, {'timestamp': current_time})
            return True

        transaction = firestore_client.transaction()
        should_trigger = update_last_trigger_time(transaction)
        if not should_trigger:
            return

        # Trigger the Cloud Build job since the cooldown period has passed
        trigger_cloud_build()
        logger.info("Cloud Build job triggered successfully.")

    except Exception as e:
        log_error(logger, e, 'Cloud Build Trigger Function')
        raise


def deploy_cloud_function(project_id, region, function_name, entry_point, runtime, trigger_topic, env_vars):
    """
    Deploy a Cloud Function using the client library.
    """
    try:
        log_step(logger, 'Deploying Cloud Function', 'Deployment')

        client = functions_v1.CloudFunctionsServiceClient()
        parent = f'projects/{project_id}/locations/{region}'

        # Prepare the Cloud Function source code zip file
        # Assuming the function code is in a directory named 'cloud_function_code'
        source_archive_url = f'gs://{project_id}-cloud-functions/{function_name}.zip'
        # You need to upload the zip file to the specified GCS bucket

        function = functions_v1.CloudFunction(
            name=f'{parent}/functions/{function_name}',
            entry_point=entry_point,
            runtime=runtime,
            environment_variables=env_vars,
            event_trigger=functions_v1.EventTrigger(
                event_type='google.pubsub.topic.publish',
                resource=f'projects/{project_id}/topics/{trigger_topic}',
                retry_policy=functions_v1.EventTrigger.RetryPolicy.RETRY_POLICY_RETRY
            ),
            source_archive_url=source_archive_url,
            service_account_email=f'{project_id}@appspot.gserviceaccount.com', 
            ingress_settings=functions_v1.CloudFunction.IngressSettings.ALLOW_ALL
        )

        operation = client.create_function(request={'location': parent, 'function': function})
        response = operation.result()

        if response.status == functions_v1.CloudFunctionStatus.ACTIVE:
            logger.info(f"Cloud Function '{function_name}' deployed successfully.")
        else:
            logger.error(f"Cloud Function '{function_name}' deployment failed with status: {response.status}")

    except Exception as e:
        log_error(logger, e, 'Deploying Cloud Function')
        raise

def setup_cloud_run(project_id, service_name, image_url, region):
    """
    Set up a Cloud Run service.
    """
    try:
        log_step(logger, 'Setting up Cloud Run Service', 'Deployment')

        client = run_v2.ServicesClient()

        service = run_v2.Service()
        service.template = run_v2.RevisionTemplate()
        service.template.containers = [
            run_v2.Container(
                image=image_url,
                env_vars=[run_v2.EnvVar(name="ENV_VAR", value="production")],  # Optional env vars
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
        logger.info(f"Cloud Run service created: {response.name}")
        return response

    except Exception as e:
        log_error(logger, e, 'Setting up Cloud Run Service')
        raise

def main():
    parser = argparse.ArgumentParser(description='Deploy to Vertex AI, set up Cloud Build triggers, and configure CI/CD with Cloud Run and Pub/Sub for continuous training')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model_name', required=True, help='Name of the machine learning model')
    parser.add_argument('--model_path', required=True, help='Path to the model artifacts')
    parser.add_argument('--endpoint_name', required=True, help='Name for the Vertex AI endpoint')
    parser.add_argument('--repo_name', required=True, help='GitHub repository name')
    parser.add_argument('--branch_name', required=True, help='GitHub branch name')
    parser.add_argument('--service_name', required=True, help='Cloud Run service name')
    parser.add_argument('--image_url', required=True, help='Docker image URL for Cloud Run')
    parser.add_argument('--region', required=True, help='GCP region for deployment')
    parser.add_argument('--storage_bucket', required=False, help='Cloud Storage bucket to monitor for new data')
    parser.add_argument('--cooldown_period', type=int, default=300, help='Cooldown period in seconds between Cloud Build jobs')
    parser.add_argument('--trigger_id', required=True, help='Cloud Build trigger ID for retraining jobs')
    parser.add_argument('--notification_channel', required=True, help='Notification channel ID for build status notifications')
    parser.add_argument('--canary_traffic_percent', type=int, default=10, help='Canary traffic split percentage')

    args = parser.parse_args()

    try:
        # Step 1: Deploy the model to Vertex AI
        logger.info(f"Deploying model '{args.model_name}' to Vertex AI...")
        endpoint_resource_name, model_resource_name = deploy_to_vertex_ai(
            project_id=args.project_id,
            model_path=args.model_path,
            endpoint_name=args.endpoint_name,
            model_name=args.model_name,
            canary_traffic_percent=args.canary_traffic_percent
        )

        # Step 2: Set up Cloud Build trigger
        logger.info("Setting up Cloud Build trigger for continuous training...")
        trigger_response = setup_cloud_build_trigger(
            project_id=args.project_id,
            repo_name=args.repo_name,
            branch_name=args.branch_name,
            storage_bucket=args.storage_bucket
        )

        # Step 3: Deploy the Cloud Function for cooldown (Pub/Sub)
        logger.info("Setting up Pub/Sub topic and deploying Cloud Function for cooldown mechanism...")

        # Ensure the Pub/Sub topic exists
        pubsub_client = pubsub_v1.PublisherClient()
        topic_path = pubsub_client.topic_path(args.project_id, 'cloud-build-trigger')
        try:
            pubsub_client.get_topic(request={"topic": topic_path})
            logger.info(f"Pub/Sub topic '{topic_path}' already exists.")
        except pubsub_client.exceptions.NotFound:
            pubsub_client.create_topic(name=topic_path)
            logger.info(f"Created Pub/Sub topic '{topic_path}'.")

        # Deploy Cloud Function for cooldown
        deploy_cloud_function(
            project_id=args.project_id,
            region=args.region,
            function_name='cloud_build_trigger',
            entry_point='cloud_build_trigger',
            runtime='python39',
            trigger_topic='cloud-build-trigger',
            env_vars={
                'PROJECT_ID': args.project_id,
                'TRIGGER_ID': args.trigger_id,
                'COOLDOWN_PERIOD': str(args.cooldown_period)
            }
        )

        # Step 4: Set up Cloud Run service for deployment
        logger.info("Setting up Cloud Run service for deployment...")
        service_response = setup_cloud_run(
            project_id=args.project_id,
            service_name=args.service_name,
            image_url=args.image_url,
            region=args.region
        )

        # Output results
        logger.info(f"Deployment to Vertex AI completed. Endpoint: {endpoint_resource_name}, Model: {model_resource_name}")
        logger.info(f"Cloud Build trigger '{trigger_response.name}' created.")
        logger.info(f"Cloud Run service '{service_response.name}' created.")
        logger.info("MLOps pipeline with Cloud Build, Pub/Sub, Cloud Function cooldown, and Cloud Run setup completed successfully.")

    except Exception as e:
        log_error(logger, e, 'Main Execution')
        raise


if __name__ == '__main__':
    main()
    

