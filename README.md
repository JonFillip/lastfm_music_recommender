# LastFM Music Recommendation System

This project implements an end-to-end machine learning pipeline for music recommendation using Kubeflow and Vertex AI.

## Project Structure

```
lastfm_music_recommendation/
├── configs/
│   └── pipeline_config.yaml
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
   git clone https://github.com/your-username/spotify_music_recommendation.git
   cd spotify_music_recommendation
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Kubeflow and Vertex AI according to their respective documentation.

## Running the Pipeline

1. Update the `configs/pipeline_config.yaml` file with your desired configuration.

2. Compile the Kubeflow pipeline:
   ```
   python kubeflow/pipeline.py
   ```

3. Upload the generated `music_recommendation_pipeline.yaml` file to your Kubeflow Pipelines UI or use the Kubeflow Pipelines SDK to run the pipeline.

## Development

- Use the `notebooks/exploratory_analysis.ipynb` for data exploration and prototyping.
- Implement new features or algorithms in the `src/` directory.
- Add tests to the `tests/` directory and run them regularly.

## Monitoring

The pipeline includes monitoring components that track model performance and data drift. Access these metrics through the Kubeflow Pipelines UI or the configured monitoring tools.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
