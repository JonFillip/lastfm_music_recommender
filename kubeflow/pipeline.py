from kfp import dsl
from kfp.components import load_component_from_file

# Load components
data_ingestion_op = load_component_from_file("components/data_ingestion/component.yaml")
data_preprocessing_op = load_component_from_file("components/preprocess/component.yaml")
feature_engineering_op = load_component_from_file("components/feature_engineering/component.yaml")
feature_store_op = load_component_from_file("components/feature_store/component.yaml")
hyperparameter_tuning_op = load_component_from_file("components/hyperparameter_tuning/component.yaml")
model_training_op = load_component_from_file("components/train/component.yaml")
model_evaluation_op = load_component_from_file("components/evaluation/component.yaml")
model_deployment_op = load_component_from_file("components/deploy/component.yaml")
model_monitoring_op = load_component_from_file("components/monitoring/component.yaml")

@dsl.pipeline(
    name='LastFM Music Recommender Pipeline',
    description='End-to-end ML pipeline for music recommendation'
)
def lastfm_music_recommender_pipeline(
    project_id: str,
    region: str,
    bucket_name: str,
    data_version: str,
    model_name: str,
    endpoint_name: str,
    feature_store_id: str,
    entity_type_id: str,
    min_accuracy: float = 0.8,
    max_training_time: int = 7200,
    monitoring_interval: int = 3600
):
    output_path = f'gs://{bucket_name}/data/raw/top_tracks_{data_version}.csv'
    train_path = f'gs://{bucket_name}/data/processed/train_{data_version}.csv'
    val_path = f'gs://{bucket_name}/data/processed/val_{data_version}.csv'
    test_path = f'gs://{bucket_name}/data/processed/test_{data_version}.csv'

    with dsl.ExitHandler(exit_op=model_monitoring_op(
        project_id=project_id,
        model_name=model_name,
        endpoint_name=endpoint_name,
        monitoring_interval=monitoring_interval
    )):
        data_ingestion_task = data_ingestion_op(
            project_id=project_id,
            output_path=output_path
        ).set_cpu_limit('1').set_memory_limit('2G')
        
        preprocess_task = data_preprocessing_op(
            input_data=data_ingestion_task.outputs['output_data'],
            output_train_path=train_path,
            output_val_path=val_path,
            output_test_path=test_path
        ).set_cpu_limit('2').set_memory_limit('4G')
        
        feature_engineering_task = feature_engineering_op(
            input_train_data=preprocess_task.outputs['train_data'],
            input_val_data=preprocess_task.outputs['val_data'],
            input_test_data=preprocess_task.outputs['test_data']
        ).set_cpu_limit('2').set_memory_limit('4G')
        
        feature_store_task = feature_store_op(
            project_id=project_id,
            region=region,
            feature_store_id=feature_store_id,
            entity_type_id=entity_type_id,
            engineered_features=feature_engineering_task.outputs['train_data']
        ).set_cpu_limit('2').set_memory_limit('4G')
        
        hp_tuning_task = hyperparameter_tuning_op(
            project_id=project_id,
            train_data=feature_store_task.outputs['feature_store_uri'],
            val_data=feature_engineering_task.outputs['val_data'],
            max_training_time=max_training_time
        ).set_gpu_limit(1)
        
        train_task = model_training_op(
            project_id=project_id,
            train_data=feature_store_task.outputs['feature_store_uri'],
            val_data=feature_engineering_task.outputs['val_data'],
            hp_params=hp_tuning_task.outputs['best_hyperparameters'],
            model_name=model_name
        ).set_gpu_limit(1)
        
        evaluate_task = model_evaluation_op(
            project_id=project_id,
            model=train_task.outputs['model'],
            test_data=feature_engineering_task.outputs['test_data'],
            min_accuracy=min_accuracy
        ).set_cpu_limit('2').set_memory_limit('4G')
        
        with dsl.Condition(evaluate_task.outputs['accuracy'] >= min_accuracy):
            deploy_task = model_deployment_op(
                project_id=project_id,
                model=train_task.outputs['model'],
                model_name=model_name,
                endpoint_name=endpoint_name,
                region=region
            )
        
        # Set the order of execution
        preprocess_task.after(data_ingestion_task)
        feature_engineering_task.after(preprocess_task)
        feature_store_task.after(feature_engineering_task)
        hp_tuning_task.after(feature_store_task)
        train_task.after(hp_tuning_task)
        evaluate_task.after(train_task)

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(lastfm_music_recommender_pipeline, 'lastfm_music_recommender_pipeline.yaml')