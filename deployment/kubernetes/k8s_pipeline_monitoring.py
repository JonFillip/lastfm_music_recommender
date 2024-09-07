import time
import psutil
import threading
import GPUtil
import os
from abc import ABC, abstractmethod
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from src.utils.logging_utils import setup_logger, log_error

logger = setup_logger('pipeline_monitoring')

class MonitoringSystem(ABC):
    @abstractmethod
    def record_metric(self, metric_name, value, labels=None):
        pass

    @abstractmethod
    def monitor_system_resources(self):
        pass

    @abstractmethod
    def record_request_latency(self, start_time):
        pass

    @abstractmethod
    def monitor_drift(self, data_drift, prediction_drift):
        pass

    @abstractmethod
    def start_server(self):
        pass

class PrometheusMonitoring(MonitoringSystem):
    def __init__(self):
        self.pipeline_duration = Gauge('pipeline_duration_seconds', 'Duration of the pipeline run')
        self.step_duration = Gauge('step_duration_seconds', 'Duration of each pipeline step', ['step'])
        self.model_accuracy = Gauge('model_accuracy', 'Accuracy of the trained model')
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.gpu_usage = Gauge('gpu_usage_percent', 'GPU usage percentage')
        self.memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
        self.cpu_memory_usage = Gauge('cpu_memory_usage_percent', 'CPU memory usage percentage')
        self.gpu_memory_usage = Gauge('gpu_memory_usage_percent', 'GPU memory usage percentage')
        self.request_latency = Histogram('request_latency_seconds', 'Latency of request-response cycles')
        self.request_count = Counter('request_count', 'Number of requests processed')
        self.prediction_drift = Gauge('prediction_drift', 'Prediction drift metric')
        self.data_drift = Gauge('data_drift', 'Data drift metric')

    def record_metric(self, metric_name, value, labels=None):
        metric = getattr(self, metric_name, None)
        if metric:
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        else:
            logger.warning(f"Metric {metric_name} not found")

    def monitor_system_resources(self):
        while True:
            try:
                self.record_metric('cpu_usage', psutil.cpu_percent())
                self.record_metric('memory_usage', psutil.virtual_memory().percent)
                self.record_metric('cpu_memory_usage', psutil.virtual_memory().used / psutil.virtual_memory().total * 100)

                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_avg_load = sum(gpu.load for gpu in gpus) / len(gpus) * 100
                    gpu_avg_memory = sum(gpu.memoryUsed for gpu in gpus) / sum(gpu.memoryTotal for gpu in gpus) * 100
                    self.record_metric('gpu_usage', gpu_avg_load)
                    self.record_metric('gpu_memory_usage', gpu_avg_memory)
                else:
                    self.record_metric('gpu_usage', 0)
                    self.record_metric('gpu_memory_usage', 0)

                time.sleep(60)  # Update every 60 seconds
            except Exception as e:
                log_error(logger, e, 'System Resource Monitoring')

    def record_request_latency(self, start_time):
        latency = time.time() - start_time
        self.request_latency.observe(latency)
        self.request_count.inc()
        return latency

    def monitor_drift(self, data_drift, prediction_drift):
        self.record_metric('data_drift', data_drift)
        self.record_metric('prediction_drift', prediction_drift)

    def start_server(self, port=8000):
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")

def main(monitoring_system):
    try:
        monitoring_system.start_server()

        threading.Thread(target=monitoring_system.monitor_system_resources, daemon=True).start()

        pipeline_start = time.time()

        steps = ['data_ingestion', 'data_validation', 'data_preprocessing', 'model_training', 'model_evaluation']
        for step in steps:
            step_start = time.time()
            time.sleep(5)  # Replace with actual step execution
            step_duration = time.time() - step_start
            monitoring_system.record_metric('step_duration', step_duration, labels={'step': step})

        pipeline_duration = time.time() - pipeline_start
        monitoring_system.record_metric('pipeline_duration', pipeline_duration)

        model_accuracy = 0.85  # Replace with actual model accuracy
        monitoring_system.record_metric('model_accuracy', model_accuracy)

        # Simulate drift monitoring
        data_drift = 0.05  # Replace with actual data drift metric
        prediction_drift = 0.03  # Replace with actual prediction drift metric
        monitoring_system.monitor_drift(data_drift, prediction_drift)

        for _ in range(10):  # Simulate 10 requests
            req_start = time.time()
            time.sleep(0.1)  # Replace with actual request processing time
            latency = monitoring_system.record_request_latency(req_start)
            logger.info(f"Request processed with latency: {latency} seconds")

        logger.info("Pipeline monitoring completed")

    except Exception as e:
        log_error(logger, e, 'Pipeline Monitoring')

if __name__ == "__main__":
    monitoring_platform = os.getenv('MONITORING_PLATFORM', 'prometheus')
    if monitoring_platform == 'prometheus':
        monitoring_system = PrometheusMonitoring()
    else:
        raise ValueError(f"Unsupported monitoring platform: {monitoring_platform}")
    
    main(monitoring_system)

# To extend this system for other platforms (e.g., CloudWatch for AWS):
# 1. Create a new class that inherits from MonitoringSystem (e.g., CloudWatchMonitoring)
# 2. Implement all the abstract methods in the new class
# 3. Add the new monitoring system to the if-else block in the __main__ section
# Example:
# elif monitoring_platform == 'cloudwatch':
#     monitoring_system = CloudWatchMonitoring()