### Import packages for data manipulation

import pandas as pd
import numpy as np
import re

### Import packages for feature extraction

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher


### Determining similarity of question and answer
class JaccardSimilarity(BaseEstimator, TransformerMixin):

    def __init__(self, df):
        pass
    
    def transform(self, df):
        
        ### Define Jaccard Similarity function
        def get_jaccard_sim(column1, column2): 
            a = set(column1.split()) 
            b = set(column2.split())
            c = a.intersection(b)
            return float(len(c)) / (len(a) + len(b) - len(c))
            
        ### Calculate similarity score between question and answer
        df_new = df[['answer_text_clean', 'question_text_clean']].copy()
        df_new['jaccard_similarity_score'] = df_new.apply(lambda x: get_jaccard_sim(str(x['answer_text_clean']), str(x['question_text_clean'])), axis = 1)
        ### Drop text
        df_new = df_new.drop(columns = ['answer_text_clean', 'question_text_clean'], axis = 1)
        return df_new
    
    def fit(self, df, y=None):
        ### Unless error returns self
        return self


### Determining similarity of question and answer
class Similarity(BaseEstimator, TransformerMixin):

    def __init__(self, df):
        pass
    
    def transform(self, df):
        
        ### Define similarity function
        def similar(column1, column2):
            return SequenceMatcher(None, column1, column2).ratio()
        ### Calculate similarity score between question and answer
        df_new = df[['answer_text_clean', 'question_text_clean']].copy()
        df_new['similarity_score'] = df_new.apply(lambda x: similar(str(x['answer_text_clean']), str(x['question_text_clean'])), axis = 1)
        ### Drop text
        df_new = df_new.drop(columns = ['answer_text_clean', 'question_text_clean'], axis = 1)
        return df_new
    
    def fit(self, df, y=None):
        ### Unless error returns self
        return self

### Count number of words in an answer

class WordCounter(BaseEstimator, TransformerMixin):

    def __init__(self, df):
        pass

    def transform(self, df):
        ### Variable name to compute number of words on
        name = df.columns
        ### Make into list
        answer_list = df[name[0]].tolist()
        ### Compute number of words for each answer
        wordcount = [len(re.findall(r'\w+', str(answer))) for answer in answer_list]
        ### Make into a pandas df
        df_new = pd.DataFrame(wordcount)
        ### Add suffix
        df_new = df_new.add_suffix(name)
        return df_new

    def fit(self, df, y=None):
        ### Unless error returns self
        return self


### Determining whether or not answer contains code

class CodeCheck(BaseEstimator, TransformerMixin):

    def __init__(self, df):
        pass
    
    def transform(self, df):
        ### Check if answer contains code or not
        df_new = df[['answer_text']].copy()
        df_new['code_binary'] = df_new['answer_text'].str.contains('<code>', regex=False)*1      
        ### Drop text
        df_new = df_new.drop(columns = ['answer_text'], axis = 1)
        return df_new
    
    def fit(self, df, y=None):
        ### Unless error returns self
        return self


### Determining whether or not answer contains code

class CodeCounter(BaseEstimator, TransformerMixin):

    def __init__(self, df):
        pass
    
    def transform(self, df):
        ### Check if answer contains code or not
        df_new = df[['answer_text']].copy()
        df_new['code_count'] = df_new['answer_text'].str.count('<code>')     
        ### Drop text
        df_new = df_new.drop(columns = ['answer_text'], axis = 1)
        return df_new
    
    def fit(self, df, y=None):
        ### Unless error returns self
        return self


### Compute n grams from a dataframe for a given variable

class Ngrams(BaseEstimator, TransformerMixin):

    def __init__(self, df):
        pass

    def transform(self, df):
        ### Save name of variable to analyze
        name = df.columns
        #### Initiate TfidfVectorizer
        vectorizer = TfidfVectorizer(strip_accents = 'unicode', use_idf = True, \
                                     stop_words = 'english', analyzer = 'word', \
                                     ngram_range = (1, 2) , max_features = 300)

        ### Fit to data
        X_train = vectorizer.fit_transform(df[name[0]].values.astype(str))
        # X_train = X_train.toarray()
        # is this needed? how do I address mismatching shape problem

        ### Return sparse matrix
        return X_train
    
    def fit(self, df, y=None):
        ### Unless error returns self
        return self



