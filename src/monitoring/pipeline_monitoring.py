from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
import psutil
import threading
import GPUtil
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

# Vertex AI monitoring client
client = monitoring_v3.MetricServiceClient()
project_name = client.common_project_path("your-project-id")  # Replace with your GCP project ID

def record_pipeline_duration(duration):
    PIPELINE_DURATION.set(duration)

def record_step_duration(step, duration):
    STEP_DURATION.labels(step=step).set(duration)

def record_model_accuracy(accuracy):
    MODEL_ACCURACY.set(accuracy)

def monitor_system_resources():
    while True:
        try:
            CPU_USAGE.set(psutil.cpu_percent())
            MEMORY_USAGE.set(psutil.virtual_memory().percent)
            CPU_MEMORY_USAGE.set(psutil.virtual_memory().used / psutil.virtual_memory().total * 100)

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_avg_load = sum(gpu.load for gpu in gpus) / len(gpus) * 100
                gpu_avg_memory = sum(gpu.memoryUsed for gpu in gpus) / sum(gpu.memoryTotal for gpu in gpus) * 100
                GPU_USAGE.set(gpu_avg_load)
                GPU_MEMORY_USAGE.set(gpu_avg_memory)
            else:
                GPU_USAGE.set(0)
                GPU_MEMORY_USAGE.set(0)

            time.sleep(60)  # Update every 60 seconds
        except Exception as e:
            log_error(logger, e, 'System Resource Monitoring')

def start_prometheus_server(port=8000):
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")

def create_vertex_ai_metric(metric_name, value):
    try:
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/{metric_name}"
        series.resource.type = "global"
        point = series.points.add()
        point.value.double_value = value
        now = time.time()
        point.interval.end_time.seconds = int(now)
        point.interval.end_time.nanos = int((now - int(now)) * 10**9)

        client.create_time_series(request={"name": project_name, "time_series": [series]})
        logger.info(f"Metric {metric_name} with value {value} sent to Vertex AI")
    except Exception as e:
        log_error(logger, e, 'Vertex AI Metric Creation')

def record_request_latency(start_time):
    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)
    REQUEST_COUNT.inc()  # Increment the request counter
    return latency

def main():
    try:
        # Start Prometheus server
        start_prometheus_server()

        # Start system resource monitoring in a separate thread
        threading.Thread(target=monitor_system_resources, daemon=True).start()

        # Simulating pipeline execution and monitoring
        pipeline_start = time.time()

        # Simulate pipeline steps
        steps = ['data_ingestion', 'data_validation', 'data_preprocessing', 'model_training', 'model_evaluation']
        for step in steps:
            step_start = time.time()
            # Simulating step execution
            time.sleep(5)  # Replace with actual step execution
            step_duration = time.time() - step_start
            record_step_duration(step, step_duration)
            create_vertex_ai_metric(f"{step}_duration", step_duration)

        pipeline_duration = time.time() - pipeline_start
        record_pipeline_duration(pipeline_duration)
        create_vertex_ai_metric("pipeline_duration", pipeline_duration)

        # Simulate model accuracy
        model_accuracy = 0.85  # Replace with actual model accuracy
        record_model_accuracy(model_accuracy)
        create_vertex_ai_metric("model_accuracy", model_accuracy)

        # Example of monitoring request-response latency
        for _ in range(10):  # Simulate 10 requests
            req_start = time.time()
            # Simulate request processing
            time.sleep(0.1)  # Replace with actual request processing time
            latency = record_request_latency(req_start)
            logger.info(f"Request processed with latency: {latency} seconds")

        logger.info("Pipeline monitoring completed")

    except Exception as e:
        log_error(logger, e, 'Pipeline Monitoring')

if __name__ == "__main__":
    main()