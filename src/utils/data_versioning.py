from google.cloud import bigquery
from datetime import datetime

def version_dataset(project_id, source_dataset_id, source_table_id, target_dataset_id):
    client = bigquery.Client(project=project_id)
    
    # Create a new table name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_table_id = f"{source_table_id}_v{timestamp}"
    
    # Construct the query to copy data
    query = f"""
    CREATE OR REPLACE TABLE `{project_id}.{target_dataset_id}.{target_table_id}`
    AS SELECT * FROM `{project_id}.{source_dataset_id}.{source_table_id}`
    """
    
    # Run the query
    job = client.query(query)
    job.result()  # Wait for the job to complete
    
    print(f"Dataset versioned: {target_dataset_id}.{target_table_id}")
    return f"{target_dataset_id}.{target_table_id}"

# Use this function after each major data processing step
# version_dataset(project_id, 'source_dataset', 'source_table', 'versioned_dataset')