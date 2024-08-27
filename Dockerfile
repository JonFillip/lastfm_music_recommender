# Use an official Python runtime as a parent image
FROM python:3.9.0

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME myenv

# Run app.py when the container launches
CMD ["python", "kubeflow/scripts/run_pipeline.py"]
