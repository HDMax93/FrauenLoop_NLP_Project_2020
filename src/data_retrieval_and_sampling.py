### Import packages for authentication

from google.cloud import bigquery
from google.oauth2 import service_account

### Import packages for converting query results into dataframe

import pandas as pd 

## Import packages to create absolute file path &  make code independent of operating system

from pathlib import Path
import os.path

### Import packages to visualize data

import matplotlib.pyplot as plt
import seaborn as sns


### Google BigQuery Authentication 

base_path = Path("__file__").parent
full_path = (base_path / "../data/raw/GoogleBigQuery_key.json").resolve()

credentials = service_account.Credentials.from_service_account_file(os.path.join(full_path))

### Construct a BigQuery client object

client = bigquery.Client(credentials=credentials, 
project = credentials.project_id)

### Overview of Stackoverflow tables

stackoverflow = client.dataset('stackoverflow', project= 'bigquery-public-data')
print([x.table_id for x in client.list_tables(stackoverflow)])

### Make an API request

query = """
SELECT
      pq.score as question_score, pa.score as answer_score, pq.id as question_id, pa.parent_id as question_id_check, pq.title as question_title, pq.body       as question_text, pq.answer_count, pq.comment_count, pq.creation_date, pq.tags, pq.view_count, pa.body as answer_text
FROM `bigquery-public-data.stackoverflow.posts_questions` pq
INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` pa ON pq.id = pa.parent_id
WHERE pa.creation_date BETWEEN "2019-05-30 00:00:00.000 UTC" AND "2020-05-30 00:00:00.000 UTC"
"""

dataframe = (
    client.query(query)
    .result()
    .to_dataframe()
)

len(dataframe)


### Save dataframe to a csv file

base_path = Path("__file__").parent
full_path = (base_path / "../data/raw/stackoverflow_raw.csv").resolve()

dataframe.to_csv(os.path.join(full_path))

### Display query results

print(dataframe)

### Check for null values

dataframe.isnull().values.any()
dataframe.isnull().sum()

#### Histogram of distribution of the Stackoverflow answer scores

f, ax = plt.subplots(figsize=(60,60))
sns.countplot(x='answer_score', data=dataframe)
plt.xlim(None, 100) 
plt.ylim(0, 9000) 

base_path = Path("__file__").parent
full_path = (base_path / "../reports/figures/stackoverflow_answerscore_distribution.png").resolve()
plt.savefig(full_path, dpi=300)

plt.show()

#### Histogram of distribution of the number of views of the question & answer

f, ax = plt.subplots(figsize=(60,60))
sns.countplot(x='view_count', data=dataframe)
plt.xlim(None, 200) 
plt.ylim(0, None) 

base_path = Path("__file__").parent
full_path = (base_path / "../reports/figures/stackoverflow_viewcount_distribution.png").resolve()
plt.savefig(full_path, dpi=300)

plt.show()

### Summary of numeric variables

dataframe.describe()

# mean answer score is 0.83 while 50% percentile / median score is 0.
# 25th percentile for view count is 39 views, median is 64 views;


### Drop all datarows with answers that have gotten fewer than 39 views (25th percentile in view_count distribution)

dataframe_views = dataframe.drop(dataframe[dataframe.view_count < 39].index)
len(dataframe_views)

### Save dataframe with only > 39 views to a csv file

base_path = Path("__file__").parent
full_path = (base_path / "../data/raw/stackoverflow_raw_views.csv").resolve()

dataframe_views.to_csv(os.path.join(full_path))

### Assign category (bad, good, great) for score; ignore zero scores 

score_bucketing_all = lambda x: 'bad' if x < 0 else 'good' if (x >= 1 and x <= 6) else 'great' if x >= 7 else None

dataframe_views['score_cat_all'] = dataframe_views['answer_score'].apply(score_bucketing_all)

print("{} unique values in column".format('score_cat_all'))
print("{}".format(dataframe_views['score_cat_all'].unique()),"\n")

### Look at dataset

print(dataframe_views)

### Look at the data to understand the type and missing values

print(dataframe_views.info())

### Sample equal amounts of data rows for "bad", "good" and "great" answers from main dataset

dataframe_sampled_all = dataframe_views.groupby('score_cat_all', group_keys=False).apply(lambda x: x.sample(n=13000, random_state = 1)).reset_index(drop=True)

### Check if equal sampling of "bad", "good", and "great" answers was successful

dataframe_sampled_all.score_cat_all.value_counts()

### Save dataframe to a csv file

base_path = Path("__file__").parent
full_path = (base_path / "../data/raw/stackoverflow_raw_views_sampled_all.csv").resolve()

dataframe_sampled_all.to_csv(os.path.join(full_path))

### Sample only one answer randomly per stackoverflow question

df_all_final = dataframe_sampled_all.groupby('question_id', group_keys=False).apply(lambda x: x.sample(n = 1, random_state = 1)).reset_index(drop=True)

### Sample equal amounts of data rows for "bad", "good" and "great" answers from main dataset with only one answer per question

df_all_final = df_all_final.groupby('score_cat_all', group_keys=False).apply(lambda x: x.sample(n=10000, random_state = 1)).reset_index(drop=True)

### Check if classes "bad", "good", and "great" are balanced

df_all_final.score_cat_all.value_counts()

### Save sample of dataframe to a csv file

base_path = Path("__file__").parent
full_path = (base_path / "../data/raw/stackoverflow_raw_finalsample_all.csv").resolve()
df_all_final.to_csv(os.path.join(full_path))