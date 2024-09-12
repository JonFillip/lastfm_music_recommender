# LastFM Music Recommendation System

This project implements an end-to-end machine learning pipeline for music recommendation using Kubeflow and Vertex AI.

## Project Structure

```
lastfm_music_recommendation/
├── configs/
│   └── pipeline_config.yaml
├── deployment/
│   └── vertex_ai/
│       ├── vertex_deployment.py
│       └── vertex_ai_monitoring.py
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
├── scripts/
│   └── run_pipeline.sh
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
│   ├── monitoring/
│   │   └── pipeline_monitoring.py
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
├── cloudbuild.yaml
└── requirements.txt
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
   git clone https://github.com/JonFillip/lastfm_music_recommender.git
   cd lastfm_music_recommendation
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Kubeflow and Vertex AI according to their respective documentation.

4. Make the run script executable:
   ```
   chmod +x scripts/run_pipeline.sh
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
   ./scripts/run_pipeline.sh
   ```

This script will:
- Compile the Kubeflow pipeline
- Submit the pipeline for execution
- Wait for the pipeline to complete
- Deploy the model to Vertex AI
- Set up Vertex AI monitoring (Start the Prometheus monitoring server if deploying to a Kubernetes environment)

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

The project includes a comprehensive monitoring system that tracks model performance, system resources, and data drift. There are two main components to the monitoring system:

### 1. Vertex AI Monitoring (deployment/vertex_ai/vertex_ai_monitoring.py)

This component sets up monitoring and alerts specifically for models deployed on Vertex AI.

Key features:
- Monitors efficiency metrics: CPU utilization, GPU utilization, memory utilization, service latency, throughput, and error rate.
- Logs sample request-response payloads to Cloud Storage for analysis.
- Implements an automated process to compute and store serving statistics in BigQuery.
- Detects training-serving skew and data drift by comparing serving data statistics to baseline training data statistics.
- Creates custom metrics and alerts for data drift, prediction drift, resource utilization, and latency.

To set up Vertex AI monitoring:

```
python deployment/vertex_ai/vertex_ai_monitoring.py \
    --project_id your-gcp-project-id \
    --model_name your-model-name
```

### 2. Data Validation (src/data_processing/data_validation.py)

This component handles data validation, schema generation, and drift detection.

Key features:
- Generates and saves data schema to Google Cloud Storage.
- Validates both training and serving data, saving statistics to GCS.
- Compares training and serving statistics to detect anomalies.
- Visualizes statistics and saves plots.
- Detects data drift between training and serving data.

The data validation process is integrated into the main pipeline and can also be run separately for ad-hoc analysis.

### Configuring Data Drift Threshold

You can configure the data drift threshold in the `configs/pipeline_config.yaml` file. Under the `data_validation` section, set the `drift_threshold` value:

```yaml
data_validation:
  schema_path: '/data/schema/features_schema.pbtxt'
  schema_version: '1.0.0'
  drift_threshold: 0.1  # Adjust this value as needed
```

This threshold is used in both the Vertex AI monitoring and the data validation components. When the drift score for any feature exceeds this threshold, a warning will be logged, and an alert will be triggered in the Vertex AI monitoring system.

## Continuous Integration and Deployment

The project uses Cloud Build for continuous integration and Cloud Run for continuous delivery. The process is defined in the `cloudbuild.yaml` file.

Key steps in the CI/CD pipeline:
1. Trigger: Any push to the main branch initiates the pipeline.
2. Build: The Docker image is built using the project's Dockerfile.
3. Test: Unit tests are run to ensure code quality.
4. Deploy: If tests pass, the new image is deployed to Cloud Run.

To set up the CI/CD pipeline:
1. Enable the Cloud Build and Cloud Run APIs in your GCP project.
2. Configure the Cloud Build trigger to watch your repository.
3. Ensure the necessary permissions are set for Cloud Build to deploy to Cloud Run.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
