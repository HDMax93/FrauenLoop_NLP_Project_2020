import requests
import json
import csv
import pandas as pd
import random
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo


FIFTEEN_MINUTES = 900

@on_exception(expo, RateLimitException, max_tries=8)
@limits(calls=15, period=FIFTEEN_MINUTES)
def call_api(url, params):
    response = requests.get(url, params)

    if response.status_code != 200:
        raise Exception('API response: {}'.format(response.status_code))
    return response


url = "https://api.crunchbase.com/v3.1/organizations"
querystring = {"locations":"Berlin","organization_types":"company","sort_order":"created_at DESC","page":"1","user_key":"e3df6582c6b0b66ca02d192734ae8b37"}

call_api(url, params = querystring)

"""
# Load credentials from json file
with open("crunchbase_credentials.json", "r") as file:
    creds = json.load(file)
"""

"""
response = requests.request("GET", url, params=querystring)

print(response.text)
"""

""" 
--Body Request--
{
  "field_ids": [
    "permalink",
    "api_path",
    "web_path",
    "api_url",
    "name",
    "stock_exchange", 
    "stock_symbol", 
    "primary_role", 
    "short_description", 
    "profile_image_url", 
    "domain",
    "homepage_url",
    "facebook_url",
    "twitter_url",
    "linkedin_url",
    "city_name",
    "region_name",
    "country_code",
    "created_at",
    "updated_at"
  ],
  "query": [
    {
      "type": "predicate",
      "field_id": "facet_ids",
      "operator_id": "includes",
      "values": [
        "company"
      ]
    }
  ]
}

 """