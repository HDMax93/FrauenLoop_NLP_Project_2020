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

from utils.feature_helpers import Similarity, JaccardSimilarity, WordCounter, CodeCheck, CodeCounter, Ngrams

### Read in dataset

print(os.getcwd())

base_path = Path("__file__").parent
full_path = (base_path / "../../data/processed/stackoverflow_preprocessed_all.csv").resolve()
# Depending on running this in interactive shell vs. terminal, I need to include GitHub/FrauenLoop_NLP_Project_2020 in filepath or not...

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


### Save data tested on feature extraction functions to a csv file

base_path = Path("__file__").parent
full_path = (base_path / "../../data/processed/stackoverflow_modeling.csv").resolve()
stackoverflow.to_csv(os.path.join(full_path))