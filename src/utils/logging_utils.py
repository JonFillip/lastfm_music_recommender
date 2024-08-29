import logging
from google.cloud import logging as cloud_logging
import os

def setup_logger(name, log_level=logging.INFO):
    """Set up a logger that works both locally and on GCP."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Check if running on GCP
    if os.getenv('KUBERNETES_SERVICE_HOST'):
        # Use Google Cloud Logging
        client = cloud_logging.Client()
        handler = cloud_logging.handlers.CloudLoggingHandler(client)
    else:
        # Use local file logging
        handler = logging.FileHandler(f"{name}.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def log_error(logger, error, component):
    """Log an error with additional context."""
    logger.error(f"Error in {component}: {str(error)}", exc_info=True)

def log_step(logger, step, component):
    """Log the start of a pipeline step."""
    logger.info(f"Starting step: {step} in component: {component}")

def log_metric(logger, metric_name, metric_value, component):
    """Log a metric."""
    logger.info(f"Metric in {component}: {metric_name} = {metric_value}")