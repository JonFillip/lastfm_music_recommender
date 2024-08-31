# Spotify Music Recommendation System

This project implements an end-to-end machine learning pipeline for music recommendation using Kubeflow and Vertex AI.

## Project Structure

```
spotify_music_recommendation/
├── configs/
│   └── pipeline_config.yaml
├── deployment/
│   └── vertex_ai/
│       └── vertex_deployment.py
├── kubeflow/
│   ├── components/
│   │   ├── deploy/
│   │   ├── monitor/
│   │   ├── preprocess/
│   │   ├── test/
│   │   └── train/
│   └── pipeline.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── algorithms/
│   │   └── content_based.py
│   ├── data_processing/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocess.py
│   │   └── data_validation.py
│   ├── evaluation/
│   │   └── model_evaluation.py
│   ├── hyperparameter_tuning/
│   │   └── katib_tuning.py
│   ├── serving/
│   │   └── model_serving.py
│   └── utils/
│       ├── data_utils.py
│       ├── logging_utils.py
│       └── model_utils.py
├── tests/
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── run_pipeline.sh
```

## Pipeline Overview

The music recommendation pipeline consists of the following steps:

1. Data Ingestion
2. Data Validation
3. Data Preprocessing
4. Hyperparameter Tuning
5. Model Training
6. Model Evaluation
7. Model Serving

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/spotify_music_recommendation.git
   cd spotify_music_recommendation
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Kubeflow and Vertex AI according to their respective documentation.

4. Make the run script executable:
   ```
   chmod +x run_pipeline.sh
   ```

## Running the Pipeline

1. Update the `configs/pipeline_config.yaml` file with your desired configuration.

2. Set the required environment variables:
   ```
   export GCP_PROJECT_ID=your-gcp-project-id
   export GCS_BUCKET=your-gcs-bucket-name
   export REGION=your-gcp-region
   ```

3. Run the entire pipeline and deployment process:
   ```
   ./run_pipeline.sh
   ```

This script will:
- Compile the Kubeflow pipeline
- Submit the pipeline for execution
- Wait for the pipeline to complete
- Deploy the model to Vertex AI
- Set up a Cloud Build trigger for continuous integration
- Create a Cloud Run service for serving your model

## Manual Deployment

If you prefer to deploy manually or need more control over the deployment process, you can use the deployment script directly:

```
python deployment/vertex_ai/vertex_deployment.py \
    --project_id your-gcp-project-id \
    --model_path gs://your-bucket/model-artifacts \
    --endpoint_name music-recommender-endpoint \
    --repo_name spotify-music-recommendation \
    --branch_name main \
    --service_name music-recommender-service \
    --image_url gcr.io/your-project/music-recommender:latest \
    --region us-central1
```

Make sure to replace the placeholder values with your actual GCP project details, model artifact location, and desired deployment configuration.

## Development

- Use the `notebooks/exploratory_analysis.ipynb` for data exploration and prototyping.
- Implement new features or algorithms in the `src/` directory.
- Add tests to the `tests/` directory and run them regularly.

## Monitoring

The pipeline includes monitoring components that track model performance and data drift. Access these metrics through the Kubeflow Pipelines UI or the configured monitoring tools.

## Continuous Integration and Deployment

The project is set up with a CI/CD pipeline using Cloud Build and Cloud Run. Any push to the main branch will trigger a new build and deployment of the model serving application.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.