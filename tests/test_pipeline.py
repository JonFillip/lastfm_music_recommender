import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from src.data_processing.data_preprocess import preprocess_data, prepare_data
from src.data_processing.data_validation import (
    generate_schema, validate_data, compare_statistics, detect_data_drift, save_schema_to_gcs
)
from src.feature_engineering.feat_engineering import (
    engineer_basic_features, engineer_additional_features, refine_features,
    add_tag_popularity, add_similar_tracks_avg_playcount, add_interaction_features,
    add_target_encoding, refine_features_further, vectorize_all_text_features,
    create_preprocessing_pipeline
)
from src.algorithms.content_based import ContentBasedRecommender
from src.evaluation.model_evaluation import evaluate_model
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_data_validation as tfdv

def precision_at_k(y_true, y_pred, k):
    return np.mean([len(set(pred[:k]) & set(true)) / k for pred, true in zip(y_pred, y_true)])

def recall_at_k(y_true, y_pred, k):
    return np.mean([len(set(pred[:k]) & set(true)) / len(true) for pred, true in zip(y_pred, y_true)])

def f1_at_k(y_true, y_pred, k):
    p = precision_at_k(y_true, y_pred, k)
    r = recall_at_k(y_true, y_pred, k)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'name': ['Song1', 'Song2', 'Song3'],
            'artist': ['Artist1', 'Artist2', 'Artist3'],
            'playcount': [1000, 2000, 3000],
            'tags': ['rock, pop', 'jazz, blues', 'electronic, dance'],
            'similar_tracks': ['Track1, Track2', 'Track3, Track4', 'Track5, Track6']
        })
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.train_data_path = os.path.join(self.test_dir, 'train_data.csv')
        self.serving_data_path = os.path.join(self.test_dir, 'serving_data.csv')
        self.sample_data.to_csv(self.train_data_path, index=False)
        self.sample_data.to_csv(self.serving_data_path, index=False)
        
        # Mock GCS bucket and model name
        self.bucket_name = 'test-bucket'
        self.model_name = 'test-model'

    def tearDown(self):
        # Remove temporary directory and its contents
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_data_validation(self):
        # Generate schema
        schema = generate_schema(self.train_data_path, self.bucket_name, self.model_name, '1.0')
        self.assertIsInstance(schema, tfdv.types.Schema)
        
        # Validate training data
        train_stats, train_anomalies = validate_data(self.train_data_path, schema, self.bucket_name, self.model_name, 'train')
        self.assertIsInstance(train_stats, tfdv.types.DatasetFeatureStatisticsList)
        self.assertIsInstance(train_anomalies, tfdv.types.Anomalies)
        self.assertFalse(train_anomalies.anomaly_info)
        
        # Validate serving data
        serving_stats, serving_anomalies = validate_data(self.serving_data_path, schema, self.bucket_name, self.model_name, 'serving')
        self.assertIsInstance(serving_stats, tfdv.types.DatasetFeatureStatisticsList)
        self.assertIsInstance(serving_anomalies, tfdv.types.Anomalies)
        self.assertFalse(serving_anomalies.anomaly_info)
        
        # Compare statistics
        comparison_anomalies = compare_statistics(train_stats, serving_stats, schema)
        self.assertIsInstance(comparison_anomalies, tfdv.types.Anomalies)
        self.assertFalse(comparison_anomalies.anomaly_info)
        
        # Detect data drift
        drift_skew = detect_data_drift(train_stats, serving_stats, schema, 0.1)
        self.assertIsInstance(drift_skew, dict)
        
        # Test with invalid data
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'playcount'] = 'not a number'
        invalid_data_path = os.path.join(self.test_dir, 'invalid_data.csv')
        invalid_data.to_csv(invalid_data_path, index=False)
        
        _, invalid_anomalies = validate_data(invalid_data_path, schema, self.bucket_name, self.model_name, 'invalid')
        self.assertTrue(invalid_anomalies.anomaly_info)

    def test_end_to_end_pipeline(self):
        # Step 1: Data Validation
        schema = generate_schema(self.train_data_path, self.bucket_name, self.model_name, '1.0')
        train_stats, train_anomalies = validate_data(self.train_data_path, schema, self.bucket_name, self.model_name, 'train')
        self.assertFalse(train_anomalies.anomaly_info)  # Expecting no anomalies in the sample data

        # Step 2: Data Preprocessing
        preprocessed_data = preprocess_data(pd.read_csv(self.train_data_path))
        self.assertIsInstance(preprocessed_data, pd.DataFrame)
        self.assertGreater(len(preprocessed_data.columns), len(self.sample_data.columns))

        # Step 3: Feature Engineering
        df = engineer_basic_features(preprocessed_data)
        df = engineer_additional_features(df)
        df = refine_features(df)
        df = add_tag_popularity(df)
        df = add_similar_tracks_avg_playcount(df)
        df = add_interaction_features(df)
        df = add_target_encoding(df)
        df = refine_features_further(df)
        df, vectorizers = vectorize_all_text_features(df)

        # Create preprocessing pipeline
        pipeline = create_preprocessing_pipeline(df)
        
        # Apply preprocessing pipeline
        X = pipeline.fit_transform(df)

        # Step 4: Data Preparation
        X_train, X_test, y_train, y_test, _, _, scaler, mlb = prepare_data(X, self.sample_data)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)

        # Step 5: Model Training
        model = ContentBasedRecommender(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
        history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)
        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)

        # Step 6: Model Evaluation
        y_pred = model.predict(X_test)
        
        # Binary accuracy
        binary_accuracy = accuracy_score(y_test > 0, y_pred > 0)
        self.assertGreater(binary_accuracy, 0)
        
        # Cosine similarity
        cosine_sim = np.mean([cosine_similarity(yt.reshape(1, -1), yp.reshape(1, -1))[0][0] for yt, yp in zip(y_test, y_pred)])
        self.assertGreater(cosine_sim, 0)
        
        # Top-N Recommendation Metrics
        k = 5  # You can adjust this value
        y_true = [np.argsort(y)[-k:] for y in y_test]
        y_pred = [np.argsort(y)[-k:] for y in y_pred]
        
        precision = precision_at_k(y_true, y_pred, k)
        recall = recall_at_k(y_true, y_pred, k)
        f1 = f1_at_k(y_true, y_pred, k)
        
        self.assertGreater(precision, 0)
        self.assertGreater(recall, 0)
        self.assertGreater(f1, 0)

    def test_pipeline_with_mock_inputs(self):
        # Mock data validation
        def mock_validate_data(data_path, schema, bucket_name, model_name, data_type):
            stats = tfdv.types.DatasetFeatureStatisticsList()
            anomalies = tfdv.types.Anomalies()
            return stats, anomalies

        # Mock feature engineering steps
        def mock_feature_engineering(df):
            df['log_playcount'] = np.log1p(df['playcount'])
            df['num_tags'] = df['tags'].str.count(',') + 1
            df['num_similar_tracks'] = df['similar_tracks'].str.count(',') + 1
            return df

        # Mock data preparation
        def mock_prepare_data(X, original_df):
            y = np.random.rand(len(X), 5)  # Mock target variable
            return X[:80], X[80:], y[:80], y[80:], None, None, None, None

        # Test pipeline with mock inputs
        schema = generate_schema(self.train_data_path, self.bucket_name, self.model_name, '1.0')
        
        stats, anomalies = mock_validate_data(self.train_data_path, schema, self.bucket_name, self.model_name, 'train')
        self.assertFalse(anomalies.anomaly_info)

        preprocessed_data = preprocess_data(pd.read_csv(self.train_data_path))
        engineered_data = mock_feature_engineering(preprocessed_data)
        
        # Create a simple preprocessing pipeline for testing
        pipeline = create_preprocessing_pipeline(engineered_data, n_components=5)
        X = pipeline.fit_transform(engineered_data)

        X_train, X_test, y_train, y_test, _, _, _, _ = mock_prepare_data(X, self.sample_data)

        model = ContentBasedRecommender(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
        history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)

        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)

        # Test model evaluation with mock inputs
        y_pred = model.predict(X_test)
        
        binary_accuracy = accuracy_score(y_test > 0, y_pred > 0)
        self.assertGreater(binary_accuracy, 0)
        
        cosine_sim = np.mean([cosine_similarity(yt.reshape(1, -1), yp.reshape(1, -1))[0][0] for yt, yp in zip(y_test, y_pred)])
        self.assertGreater(cosine_sim, 0)

    def test_pipeline_output_artifacts(self):
        # Test if pipeline components generate expected output artifacts
        schema = generate_schema(self.train_data_path, self.bucket_name, self.model_name, '1.0')
        train_stats, _ = validate_data(self.train_data_path, schema, self.bucket_name, self.model_name, 'train')
        
        preprocessed_data = preprocess_data(pd.read_csv(self.train_data_path))
        
        # Feature engineering steps
        df = engineer_basic_features(preprocessed_data)
        df = engineer_additional_features(df)
        df = refine_features(df)
        df = add_tag_popularity(df)
        df = add_similar_tracks_avg_playcount(df)
        df = add_interaction_features(df)
        df = add_target_encoding(df)
        df = refine_features_further(df)
        df, vectorizers = vectorize_all_text_features(df)

        # Create preprocessing pipeline
        pipeline = create_preprocessing_pipeline(df)
        
        # Apply preprocessing pipeline
        X = pipeline.fit_transform(df)

        X_train, X_test, y_train, y_test, _, _, scaler, mlb = prepare_data(X, self.sample_data)

        model = ContentBasedRecommender(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
        model.fit(X_train, y_train, epochs=5, validation_split=0.2)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Test model artifact
            model_path = os.path.join(tmpdirname, 'test_model')
            model.save(model_path)
            self.assertTrue(os.path.exists(model_path))

            # Test preprocessing pipeline artifact
            pipeline_path = os.path.join(tmpdirname, 'preprocessing_pipeline.pkl')
            pd.to_pickle(pipeline, pipeline_path)
            self.assertTrue(os.path.exists(pipeline_path))

            # Test vectorizers artifact
            vectorizers_path = os.path.join(tmpdirname, 'vectorizers.pkl')
            pd.to_pickle(vectorizers, vectorizers_path)
            self.assertTrue(os.path.exists(vectorizers_path))

            # Test scaler artifact
            scaler_path = os.path.join(tmpdirname, 'scaler.pkl')
            pd.to_pickle(scaler, scaler_path)
            self.assertTrue(os.path.exists(scaler_path))

            # Test MultiLabelBinarizer artifact
            mlb_path = os.path.join(tmpdirname, 'mlb.pkl')
            pd.to_pickle(mlb, mlb_path)
            self.assertTrue(os.path.exists(mlb_path))

            # Test schema artifact
            schema_path = os.path.join(tmpdirname, 'schema.pbtxt')
            save_schema_to_gcs(schema, tmpdirname, self.model_name, '1.0')
            self.assertTrue(os.path.exists(schema_path))

            # Test statistics artifact
            stats_path = os.path.join(tmpdirname, 'train_stats.pb')
            with open(stats_path, 'wb') as f:
                f.write(train_stats.SerializeToString())
            self.assertTrue(os.path.exists(stats_path))

            # Test evaluation results artifact
            y_pred = model.predict(X_test)
            binary_accuracy = accuracy_score(y_test > 0, y_pred > 0)
            cosine_sim = np.mean([cosine_similarity(yt.reshape(1, -1), yp.reshape(1, -1))[0][0] for yt, yp in zip(y_test, y_pred)])
            
            k = 5
            y_true = [np.argsort(y)[-k:] for y in y_test]
            y_pred = [np.argsort(y)[-k:] for y in y_pred]
            precision = precision_at_k(y_true, y_pred, k)
            recall = recall_at_k(y_true, y_pred, k)
            f1 = f1_at_k(y_true, y_pred, k)
            
            results = {
                'binary_accuracy': binary_accuracy,
                'cosine_similarity': cosine_sim,
                'precision@5': precision,
                'recall@5': recall,
                'f1@5': f1
            }
            results_path = os.path.join(tmpdirname, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f)
            self.assertTrue(os.path.exists(results_path))

if __name__ == '__main__':
    unittest.main()