import argparse
from google.cloud import aiplatform
from google.cloud.devtools import cloudbuild_v1
from google.cloud import run_v2

def deploy_to_vertex_ai(project_id, model_path, endpoint_name):
    aiplatform.init(project=project_id)
    
    # Upload the model
    model = aiplatform.Model.upload(
        display_name=endpoint_name,
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest"
    )
    
    # Deploy the model to an endpoint
    endpoint = model.deploy(
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=3,
        accelerator_type=None,
        accelerator_count=None
    )
    
    print(f"Model deployed to endpoint: {endpoint.resource_name}")
    return endpoint

def setup_cloud_build_trigger(project_id, repo_name, branch_name):
    client = cloudbuild_v1.CloudBuildClient()
    
    trigger = cloudbuild_v1.BuildTrigger()
    trigger.name = f"{repo_name}-{branch_name}-trigger"
    trigger.github = cloudbuild_v1.GitHubEventsConfig(
        owner="your-github-username",
        name=repo_name,
        push=cloudbuild_v1.PushFilter(branch=branch_name)
    )
    trigger.filename = "cloudbuild.yaml"
    
    operation = client.create_build_trigger(project_id=project_id, trigger=trigger)
    result = operation.result()
    
    print(f"Cloud Build trigger created: {result.name}")
    return result

def setup_cloud_run(project_id, service_name, image_url, region):
    client = run_v2.ServicesClient()
    
    service = run_v2.Service()
    service.template = run_v2.RevisionTemplate()
    service.template.containers = [run_v2.Container(image=image_url)]
    
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
    
    args = parser.parse_args()
    
    # Deploy to Vertex AI
    endpoint = deploy_to_vertex_ai(args.project_id, args.model_path, args.endpoint_name)
    
    # Setup Cloud Build trigger
    trigger = setup_cloud_build_trigger(args.project_id, args.repo_name, args.branch_name)
    
    # Setup Cloud Run service
    service = setup_cloud_run(args.project_id, args.service_name, args.image_url, args.region)
    
    print("Deployment completed successfully!")