from kfp import dsl
from kfp.components import load_component_from_file

# Load components
data_ingestion_op = load_component_from_file("components/data_ingestion/component.yaml")
data_preprocessing_op = load_component_from_file("components/preprocess/component.yaml")
hyperparameter_tuning_op = load_component_from_file("components/hyperparameter_tuning/component.yaml")
model_training_op = load_component_from_file("components/train/component.yaml")
model_evaluation_op = load_component_from_file("components/test/component.yaml")
model_deployment_op = load_component_from_file("components/deploy/component.yaml")
model_monitoring_op = load_component_from_file("components/monitor/component.yaml")

@dsl.pipeline(
    name='LastFM Music Recommender Pipeline',
    description='End-to-end ML pipeline for music recommendation'
)
def lastfm_music_recommender_pipeline(
    output_path: str = 'gs://your-bucket/data/raw/top_tracks.csv',
    train_path: str = 'gs://your-bucket/data/processed/train.csv',
    val_path: str = 'gs://your-bucket/data/processed/val.csv',
    test_path: str = 'gs://your-bucket/data/processed/test.csv',
):
    data_ingestion_task = data_ingestion_op(output_path=output_path)
    
    preprocess_task = data_preprocessing_op(
        input_data=data_ingestion_task.outputs['output_data'],
        output_train_path=train_path,
        output_val_path=val_path,
        output_test_path=test_path
    )
    
    hp_tuning_task = hyperparameter_tuning_op(
        train_data=preprocess_task.outputs['train_data'],
        val_data=preprocess_task.outputs['val_data']
    )
    
    train_task = model_training_op(
        train_data=preprocess_task.outputs['train_data'],
        val_data=preprocess_task.outputs['val_data'],
        hp_params=hp_tuning_task.outputs['best_hyperparameters']
    )
    
    evaluate_task = model_evaluation_op(
        model=train_task.outputs['model'],
        test_data=preprocess_task.outputs['test_data']
    )
    
    deploy_task = model_deployment_op(
        model=train_task.outputs['model']
    )
    
    monitor_task = model_monitoring_op(
        model=train_task.outputs['model'],
        deploy_info=deploy_task.outputs['model_info']
    )
    
    # Set the order of execution
    preprocess_task.after(data_ingestion_task)
    hp_tuning_task.after(preprocess_task)
    train_task.after(hp_tuning_task)
    evaluate_task.after(train_task)
    deploy_task.after(evaluate_task)
    monitor_task.after(deploy_task)

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(lastfm_music_recommender_pipeline, 'lastfm_music_recommender_pipeline.yaml')