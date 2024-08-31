FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Install additional dependencies for Kubeflow and Vertex AI
RUN pip install --no-cache-dir kfp google-cloud-aiplatform

# Make sure all scripts are executable
RUN chmod +x src/*.py kubeflow/components/*/*.py

# Run tests by default
CMD ["python", "-m", "unittest", "discover", "tests"]
