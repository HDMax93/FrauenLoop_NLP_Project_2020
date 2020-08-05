### Import packages to create absolute file path & make code independent of operating system

from pathlib import Path
import os.path
import sys

import warnings
warnings.filterwarnings("ignore")

### Import packages for data manipulation

import pandas as pd
import numpy as np
import re

### Import feature extraction help functions

from common_utils.feature_helpers import JaccardSimilarity, Similarity, WordCounter, CodeCheck, CodeCounter, Ngrams, TopTagEncoder

### Read in dataset

print(os.getcwd())

base_path = Path("__file__").parent
full_path = (base_path / "../data/processed/stackoverflow_preprocessed_all.csv").resolve()
stackoverflow = pd.read_csv(os.path.join(full_path))

stackoverflow.info()

### Drop all observations / rows with any missing values in the column "answer_text_clean"

stackoverflow = stackoverflow.dropna(how='any', subset=['answer_text_clean', 'question_title_clean'])

### Print out dataset for overview

stackoverflow.head()

### Check if Similarity score class works as desired

similarity_scorer = Similarity(stackoverflow)
stackoverflow_new = similarity_scorer.transform(stackoverflow)
stackoverflow_new.head(10)

### Check if Jaccard Similarity score class works as desired

jaccard_similarity = JaccardSimilarity(stackoverflow)
stackoverflow_new = jaccard_similarity.transform(stackoverflow)
stackoverflow_new.head(10)

### Check if WordCounter class works as desired

wordcounter = WordCounter(stackoverflow[['answer_text_clean']])
stackoverflow_new = wordcounter.transform(stackoverflow[['answer_text_clean']])
stackoverflow_new.head()

### Check if CodeCheck class works as desired

codecheck = CodeCheck(stackoverflow) 
stackover_new = codecheck.transform(stackoverflow)
stackover_new.head()

### Check ratio of code vs. no code in answers

stackover_new['code_binary'].value_counts()

### Check if CodeCounter class works as desired

codecount = CodeCounter(stackoverflow) 
stack_new = codecount.transform(stackoverflow)

stack_new.head()

### Check distribution of code counts

stack_new['code_count'].value_counts().sort_index()

### Check if Ngrams classs works as desired

ngrams = Ngrams(stackoverflow['answer_text_clean'])
stackover_new = ngrams.transform(stackoverflow[['answer_text_clean']])

print(stackover_new)

### Check if TopTagsEncoder works as desired

toptagencoded = TopTagEncoder(stackoverflow)
stack_tags_new = toptagencoded.transform(stackoverflow['tag_list_clean'])

print(stack_tags_new)

from collections import Counter
tags_joined = " ".join(stackoverflow['tag_list_clean'])
tags_split = tags_joined.split()
print(tags_split)
most_common_words = [word for word, word_count in Counter(tags_split).most_common(50)]
print(most_common_words)

def toptagslist(text_column):
    tags_joined = " ".join(text_column)
    tags_split = tags_joined.split()
    most_common_words = [word for word, word_count in Counter(tags_split).most_common(50)]
    return most_common_words

top_tags = stackoverflow['tag_list_clean'].apply(toptagslist)
print(top_tags)

### Save data tested on feature extraction functions to a csv file

base_path = Path("__file__").parent
full_path = (base_path / "../data/processed/stackoverflow_modeling.csv").resolve()
stackoverflow.to_csv(os.path.join(full_path))




