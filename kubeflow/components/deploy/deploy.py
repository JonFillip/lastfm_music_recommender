from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Model,
)
from typing import NamedTuple
from deployment.vertex_ai.vertex_deployment import deploy_to_vertex_ai, setup_cloud_build_trigger, setup_cloud_run
from src.utils.logging_utils import setup_logger, log_error

logger = setup_logger('kubeflow_deploy_component')

# Define the OutputSpec NamedTuple
OutputSpec = NamedTuple('OutputSpec', [('endpoint', str), ('model', str)])

@component(
    packages_to_install=['google-cloud-aiplatform', 'google-cloud-build', 'google-cloud-run'],
    base_image='python:3.9'
)
def deploy_model(
    project_id: str,
    model_path: Input[Model],
    model_name: str,
    endpoint_name: str,
    output_val: Output[str],
    output_test: Output[str],
    repo_name: str = "",
    branch_name: str = "",
    service_name: str = "",
    image_url: str = "",
    region: str = "us-central1",
    setup_ci_cd: bool = False,
    canary_traffic_percent: int = 10
) -> OutputSpec:
    """
    Kubeflow component to deploy a model to Vertex AI with a canary strategy and rollback mechanism.
    """
    try:
        # Deploy to Vertex AI with canary traffic handling
        endpoint, model = deploy_to_vertex_ai(
            project_id=project_id,
            model_path=model_path.uri,
            endpoint_name=endpoint_name,
            model_name=model_name,
            canary_traffic_percent=canary_traffic_percent
        )
        
        # Write outputs to files
        with open(output_val.path, 'w') as f:
            f.write(endpoint)
        with open(output_test.path, 'w') as f:
            f.write(model)
            
        # Setup CI/CD if requested
        if setup_ci_cd:
            if not all([repo_name, branch_name, service_name, image_url]):
                raise ValueError("For CI/CD setup, repo_name, branch_name, service_name, and image_url must be provided.")
            
            # Setup Cloud Build trigger
            trigger = setup_cloud_build_trigger(project_id, repo_name, branch_name)
            
            # Setup Cloud Run service
            service = setup_cloud_run(project_id, service_name, image_url, region)
        
        logger.info("Deployment completed successfully!")
        return OutputSpec(endpoint=endpoint, model=model)
    
    except Exception as e:
        log_error(logger, e, 'Model Deployment')
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy model component for Kubeflow')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--model_path', required=True, help='Path to the model artifacts')
    parser.add_argument('--model_name', required=True, help='Name for the deployed model')
    parser.add_argument('--endpoint_name', required=True, help='Name for the Vertex AI endpoint')
    parser.add_argument('--setup_ci_cd', action='store_true', help='Set up CI/CD pipeline')
    parser.add_argument('--repo_name', help='GitHub repository name')
    parser.add_argument('--branch_name', help='GitHub branch name')
    parser.add_argument('--service_name', help='Cloud Run service name')
    parser.add_argument('--image_url', help='Docker image URL for Cloud Run')
    parser.add_argument('--region', default='us-central1', help='GCP region for deployment')
    parser.add_argument('--canary_traffic_percent', type=int, default=10, help='Percentage of traffic to route to the new model')
    
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
        setup_ci_cd=args.setup_ci_cd,
        canary_traffic_percent=args.canary_traffic_percent
    )
