#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if the required environment variables are set
if [ -z "$GCP_PROJECT_ID" ] || [ -z "$GCS_BUCKET" ] || [ -z "$REGION" ]; then
    echo "Please set the following environment variables:"
    echo "GCP_PROJECT_ID: Your Google Cloud Project ID"
    echo "GCS_BUCKET: The Google Cloud Storage bucket to store artifacts"
    echo "REGION: The GCP region for deployment (e.g., us-central1)"
    exit 1
fi

# Step 1: Compile the Kubeflow pipeline
echo "Compiling Kubeflow pipeline..."
python kubeflow/pipeline.py

# Step 2: Run the Kubeflow pipeline
echo "Running Kubeflow pipeline..."
# Note: You may need to adjust this command based on your Kubeflow setup
kfp run submit -e default -r music_recommendation_$(date +%Y%m%d_%H%M%S) -f music_recommendation_pipeline.yaml

# Step 3: Wait for the pipeline to complete
echo "Waiting for pipeline to complete..."
# Note: You may need to implement a waiting mechanism here, depending on your Kubeflow setup

# Step 4: Deploy the model to Vertex AI
echo "Deploying model to Vertex AI..."
python deployment/vertex_ai/vertex_deployment.py \
    --project_id $GCP_PROJECT_ID \
    --model_path gs://$GCS_BUCKET/model-artifacts \
    --endpoint_name music-recommender-endpoint \
    --repo_name spotify-music-recommendation \
    --branch_name main \
    --service_name music-recommender-service \
    --image_url gcr.io/$GCP_PROJECT_ID/music-recommender:latest \
    --region $REGION

# Step 5: Start Prometheus monitoring server (Optional)
# echo "Starting Prometheus monitoring server..."
# python src/monitoring/pipeline_monitoring.py &

# Step 6: Set up Vertex AI monitoring
echo "Setting up Vertex AI monitoring..."
python deployment/vertex_ai/vertex_ai_monitoring.py \
    --project_id $GCP_PROJECT_ID \
    --model_name music-recommender-model

echo "Pipeline execution, deployment, and monitoring setup completed successfully!"