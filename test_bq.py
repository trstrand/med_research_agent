
import os
import google.auth
from google.cloud import bigquery

def test_bq():
    auth_credentials, auth_project_id = google.auth.default()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or auth_project_id
    print(f"Using project: {project_id}")
    client = bigquery.Client(project=project_id)
    query = "SELECT * FROM `bigquery-public-data.pmc_open_access_commercial.metadata` LIMIT 1"
    try:
        query_job = client.query(query)
        results = query_job.result()
        for row in results:
            print(f"Success: {row}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_bq()
