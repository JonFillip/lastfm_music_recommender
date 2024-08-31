import time
import psutil
import threading
import GPUtil
import os
import json
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from google.cloud import monitoring_v3
from src.utils.logging_utils import setup_logger, log_error

logger = setup_logger('pipeline_monitoring')

# Prometheus metrics
PIPELINE_DURATION = Gauge('pipeline_duration_seconds', 'Duration of the pipeline run')
STEP_DURATION = Gauge('step_duration_seconds', 'Duration of each pipeline step', ['step'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Accuracy of the trained model')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
GPU_USAGE = Gauge('gpu_usage_percent', 'GPU usage percentage')
MEMORY_USAGE = Gauge('memory_usage_percent', 'Memory usage percentage')
CPU_MEMORY_USAGE = Gauge('cpu_memory_usage_percent', 'CPU memory usage percentage')
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_percent', 'GPU memory usage percentage')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency of request-response cycles')
REQUEST_COUNT = Counter('request_count', 'Number of requests processed')
PREDICTION_DRIFT = Gauge('prediction_drift', 'Prediction drift metric')
DATA_DRIFT = Gauge('data_drift', 'Data drift metric')

# Vertex AI monitoring client
client = monitoring_v3.MetricServiceClient()
project_name = client.common_project_path(os.getenv('GCP_PROJECT_ID', 'your-project-id'))

def record_metric(metric, value, platform='prometheus', labels=None):
    if platform == 'prometheus':
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)
    elif platform == 'vertex_ai':
        create_vertex_ai_metric(metric.__name__, value, labels)
    else:
        logger.warning(f"Unsupported monitoring platform: {platform}")

def create_vertex_ai_metric(metric_name, value, labels=None):
    try:
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/{metric_name}"
        series.resource.type = "global"
        if labels:
            series.metric.labels.update(labels)
        point = series.points.add()
        point.value.double_value = value
        now = time.time()
        point.interval.end_time.seconds = int(now)
        point.interval.end_time.nanos = int((now - int(now)) * 10**9)

        client.create_time_series(request={"name": project_name, "time_series": [series]})
        logger.info(f"Metric {metric_name} with value {value} sent to Vertex AI")
    except Exception as e:
        log_error(logger, e, 'Vertex AI Metric Creation')

def monitor_system_resources(platform='prometheus'):
    while True:
        try:
            record_metric(CPU_USAGE, psutil.cpu_percent(), platform)
            record_metric(MEMORY_USAGE, psutil.virtual_memory().percent, platform)
            record_metric(CPU_MEMORY_USAGE, psutil.virtual_memory().used / psutil.virtual_memory().total * 100, platform)

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_avg_load = sum(gpu.load for gpu in gpus) / len(gpus) * 100
                gpu_avg_memory = sum(gpu.memoryUsed for gpu in gpus) / sum(gpu.memoryTotal for gpu in gpus) * 100
                record_metric(GPU_USAGE, gpu_avg_load, platform)
                record_metric(GPU_MEMORY_USAGE, gpu_avg_memory, platform)
            else:
                record_metric(GPU_USAGE, 0, platform)
                record_metric(GPU_MEMORY_USAGE, 0, platform)

            time.sleep(60)  # Update every 60 seconds
        except Exception as e:
            log_error(logger, e, 'System Resource Monitoring')

def start_prometheus_server(port=8000):
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")

def record_request_latency(start_time, platform='prometheus'):
    latency = time.time() - start_time
    if platform == 'prometheus':
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.inc()
    elif platform == 'vertex_ai':
        create_vertex_ai_metric('request_latency', latency)
        create_vertex_ai_metric('request_count', 1)
    return latency

def monitor_drift(data_drift, prediction_drift, platform='prometheus'):
    record_metric(DATA_DRIFT, data_drift, platform)
    record_metric(PREDICTION_DRIFT, prediction_drift, platform)

def main(platform='prometheus'):
    try:
        if platform == 'prometheus':
            start_prometheus_server()

        threading.Thread(target=monitor_system_resources, args=(platform,), daemon=True).start()

        pipeline_start = time.time()

        steps = ['data_ingestion', 'data_validation', 'data_preprocessing', 'model_training', 'model_evaluation']
        for step in steps:
            step_start = time.time()
            time.sleep(5)  # Replace with actual step execution
            step_duration = time.time() - step_start
            record_metric(STEP_DURATION, step_duration, platform, labels={'step': step})

        pipeline_duration = time.time() - pipeline_start
        record_metric(PIPELINE_DURATION, pipeline_duration, platform)

        model_accuracy = 0.85  # Replace with actual model accuracy
        record_metric(MODEL_ACCURACY, model_accuracy, platform)

        # Simulate drift monitoring
        data_drift = 0.05  # Replace with actual data drift metric
        prediction_drift = 0.03  # Replace with actual prediction drift metric
        monitor_drift(data_drift, prediction_drift, platform)

        for _ in range(10):  # Simulate 10 requests
            req_start = time.time()
            time.sleep(0.1)  # Replace with actual request processing time
            latency = record_request_latency(req_start, platform)
            logger.info(f"Request processed with latency: {latency} seconds")

        logger.info("Pipeline monitoring completed")

    except Exception as e:
        log_error(logger, e, 'Pipeline Monitoring')

if __name__ == "__main__":
    platform = os.getenv('MONITORING_PLATFORM', 'prometheus')
    main(platform)