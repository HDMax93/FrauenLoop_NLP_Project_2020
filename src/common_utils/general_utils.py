### Import packages for data manipulation

import pandas as pd
import numpy as np
import re

### Import packages for feature extraction

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
from sklearn.preprocessing import MultiLabelBinarizer

### Determining similarity of question and answer
class JaccardSimilarity(BaseEstimator, TransformerMixin):
    """
    A class used to represent similarity of two strings

    ...

    Attributes
    ----------
    df : dataframe
        a dataframe containing two text columns to compare

    Methods
    -------
    transform
        Takes two text columns in a dataframe to return a similarity score for each row/pair of text/strings in dataframe

    """

    def __init__(self, df):
        pass
    
    def transform(self, df):
        """Creates similarity score by comparing two strings.

        Parameters
        ----------
        df : dataframe
        a dataframe containing two text columns to compare
        """

        ### Define Jaccard Similarity function
        def get_jaccard_sim(column1, column2): 
        """Calculates Jaccard-Similarity

        Parameters
        ----------
        column1 : column in dataframe
                first text/object column in dataframe
        column2: column in dataframe
                second text/object in dataframe
        """
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
    """A class used to represent similarity of two strings

    ...

    Attributes
    ----------
    df : dataframe
        a dataframe containing two text columns to compare

    Methods
    -------
    transform
        Takes two text columns in a dataframe to return a similarity score for each row/pair of text/strings in dataframe

    """
    def __init__(self, df):
        pass
    
    def transform(self, df):
        """Creates similarity score by comparing two strings.

        Parameters
        ----------
        df : dataframe
        a dataframe containing two text columns to compare
        """

        ### Define similarity function
        def similar(column1, column2):
            """Calculates Jaccard-Similarity

            Parameters
            ----------
            column1 : column in dataframe
                    first text/object column in dataframe
            column2: column in dataframe
                    second text/object in dataframe
            """

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
    """
    A class used to count the number of words in a string

    ...

    Attributes
    ----------
    df : dataframe
        a dataframe containing column with words/elements to count

    Methods
    -------
    transform
        Takes text columns in a dataframe to return its word count
    """

    def __init__(self, df):
        pass

    def transform(self, df):
        """Returns word count of a string.

        Parameters
        ----------
        df : dataframe
            a dataframe containing the column with string/text to count elements
        """

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
        """
        A class used to check if a string contains a code snippets

        ...

        Attributes
        ----------
        df : dataframe
            a dataframe containing column with text that may or may not contain code snippet

        Methods
        -------
        transform
            Takes text columns in a dataframe to returns 1 if string contains code and 0 if it does not
        """

    def __init__(self, df):
        pass
    
    def transform(self, df):
        """Returns dummy for whether or not string contains code snippets

        Parameters
        ----------
        df : dataframe
            a dataframe containing the column with string/text that may contain code snippet
        """

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
        """
        A class used to count how many code snippets a string contains

        ...

        Attributes
        ----------
        df : dataframe
            a dataframe containing column with text that may or may not contain code snippets

        Methods
        -------
        transform
            Takes text columns in a dataframe to returns the number of code snippets it contains
        """

    def __init__(self, df):
        pass
    
    def transform(self, df):
        """Returns count of how many code snippets a string contains

        Parameters
        ----------
        df : dataframe
            a dataframe containing the column with string/text that may contain code snippets
        """

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
        """
        A class that generates ngrams for top 300 features based on a text column

        ...

        Attributes
        ----------
        df : dataframe
            a dataframe containing column with text/string

        Methods
        -------
        transform
            Takes text columns in a dataframe to returns ngrams for tip 300 features in text
        """

    def __init__(self, df):
        pass

    def transform(self, df):
        """Takes text-column of dataframe and returns ngrams for top 30 features

        Parameters
        ----------
        df : dataframe
            a dataframe containing the column with string/text to generate ngrams
        """

        ### Save name of variable to analyze
        name = df.columns
        #### Initiate TfidfVectorizer
        vectorizer = TfidfVectorizer(strip_accents = 'unicode', use_idf = True, \
                                     stop_words = 'english', analyzer = 'word', \
                                     ngram_range = (1, 2) , max_features = 300)
        ### Fit to data
        X_train = vectorizer.fit_transform(df[name[0]].values.astype(str))

        ### Return sparse matrix
        return X_train
    
    def fit(self, df, y=None):
        ### Unless error returns self
        return self


### One-hot encode top 50 question tags

class TopTagEncoder(BaseEstimator, TransformerMixin):
        """
        A class that one-hot encodes based on a list of strings, by checking for existence of each string in text-column of dataframe

        ...

        Attributes
        ----------
        df : dataframe
            a dataframe containing column with text/string to one-hot encode

        Methods
        -------
        transform
            Takes text column in a dataframe to return one-hot encoded columns for list of string, assigning 1 if string is contained in a given dataframe-row
        """

    def __init__(self, df):
        pass

    def transform(self, df):
        """Returns dummy-columns based on a list of strings to be dummified and a text-column of dataframe which provides info if dummy is present (1) or not (0)

        Parameters
        ----------
        df : dataframe
            a dataframe containing the column with string/text to one-hot encode
        """

        df_new = df[['tag_list_clean']].copy()
        var_list = ['javascript', 'java', 'python', 'c#', 'php', 'android', 'html', 'c++', 'jquery', 'css', 'ios', 'mysql', 'sql', 'asp.net', 'r', 'node.js', 'arrays', 'c', 'ruby-on-rails', '.net', 'json', 'objective-c', 'sql-server', 'swift', 'angularjs', 'python-3.x', 'django', 'reactjs', 'excel', 'regex', 'angular', 'iphone', 'ruby', 'ajax', 'xml', 'linux', 'asp.net-mvc', 'vba', 'spring', 'database', 'wordpress', 'panas', 'wpf', 'string', 'laravel', 'xcode', 'windows', 'mongodb', 'vb.net', 'bash']
        for var in var_list:
            ### Create column name "has_tagname" for each tag in list
            new_var_name = "%s_%s" % ("has", var)
            ### Create dataframe column for each tag, and if original tag-column contains tag, assign 1
            df_new[new_var_name] = df_new['tag_list_clean'].str.contains(re.escape(var), regex=True)*1
        df_new = df_new.drop(columns = ['tag_list_clean'], axis = 1)
        return df_new

    def fit(self, df, y=None):
        ### Unless error returns self
        return self