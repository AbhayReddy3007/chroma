from google.cloud import bigquery
import os

# Path to your service account JSON file
SERVICE_ACCOUNT_JSON = r"C:\Users\P90022569\Downloads\service 2.json"

# Set environment variable for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_JSON

# Initialize BigQuery client
client = bigquery.Client()

# Define the query
query = """
SELECT eval_actual_assignee, assignee
FROM `prj-portfolio-ai-dev.portfolio_data.patent_discovery`
WHERE patent_number = 'US_12534530_B2'
"""

# Run the query
query_job = client.query(query)

# Fetch results
results = query_job.result()

# Print results
print("Results for patent US_12534530_B2:")
for row in results:
    print(f"eval_actual_assignee: {row.eval_actual_assignee}, assignee: {row.assignee}")
