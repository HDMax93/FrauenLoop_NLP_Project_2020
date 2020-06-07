"""
Utility functions to wrangle data
"""

import os
from typing import List

import pandas as pd


def read_data():
    fn = 'amazon_co-ecommerce_sample_short.csv'
    df = pd.read_csv(os.path.join('data', fn))
    return df


def wrangle_amazon_reviews() -> pd.DataFrame:
    df = read_data()
    reviews_df = df['customer_reviews'].str.split(' // ', n=4, expand=True)

    df['review_title'] = reviews_df[0]
    df['review_rating'] = reviews_df[1]
    df['review_date'] = reviews_df[2]
    df['review_customer'] = reviews_df[3]
    df['review_text'] = reviews_df[4]
    return df


def sample_amazon_reviews(df: pd.DataFrame, n: int = 1) -> List[str]:
    samples = df['review_text'].sample(n).to_list()
    return samples


def get_amazon_review(df: pd.DataFrame, index: int) -> str:
    review = df.loc[index]['review_text']
    return review
