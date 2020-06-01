
### Each time when running the code, set environment variables GOOGLE_APPLICATION_CREDENTIALS to the path of the JSON file that contains our service account key. 
### command line: export GOOGLE_APPLICATION_CREDENTIALS="/Users/HenriekeMax/Documents/Career_Development/GitHub/FrauenLoop_NLP_Project_2020/data/raw/GoogleBigQuery_key.json"

from google.cloud import bigquery
from google.oauth2 import service_account


credentials = service_account.Credentials.from_service_account_file("/Users/HenriekeMax/Documents/Career_Development/GitHub/FrauenLoop_NLP_Project_2020/data/raw/GoogleBigQuery_key.json")

scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])

client = bigquery.Client(credentials=credentials, 
project = credentials.project_id)

### Verifying authentication

def implicit():
    from google.cloud import bigquery

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client(credentials=credentials, project = credentials.project_id)
    
    # Make an authenticated API request
    buckets = list(client.list_buckets())
    print(buckets)

implicit()

### Import dataset


### Perform a query.

###############################

def query_stackoverflow():
    client = bigquery.Client()
    query_job = client.query("""
SELECT
      pq.id as question_id, pq.title as question_title, pq.body as question_text, pq.accepted_answer_id, pq.answer_count, pq.comment_count, pq.community_owned_date,
      pq.creation_date, pq.favorite_count, pq.last_activity_date, pq.last_edit_date, pq.last_editor_display_name, 
      pq.last_editor_user_id, pq.owner_display_name, pq.owner_user_id, pq.parent_id, pq.post_type_id, pq.score,
      pq.tags, pq.view_count,
      pa.id as answer_id, pa.body as answer_text
FROM `bigquery-public-data.stackoverflow.posts_questions` pq
INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` pa ON pq.id = pa.parent_id
WHERE pa.creation_date > "2019-05-30 00:00:00.000 UTC"
LIMIT 10""")

    results = query_job.result()  # Waits for job to complete.

print(results)

'''
### need to fix this part of the code 
    for row in results:
        print(row.name)

if __name__ == '__main__':
    query_stackoverflow()

'''