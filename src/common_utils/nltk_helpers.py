"""Utility functions to perform different text preprocessing tasks with nltk

"""

### Import packages for preprocessing-functions

import pandas as pd
import string
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re

### Cleaning the text

stopwords_nltk = nltk.corpus.stopwords.words('english')
print(stopwords_nltk)

### Create own list of stopwords building on nltk

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "can't", "'i'd", "it's", "I've"]

### Prepare lemmatizer

lemmatizer = WordNetLemmatizer()

###  Define function that takes a whole sentence and outputs the lemmatized sentence

def nltk2wn_tag(nltk_tag):
    """Convert the sentence to a list of tuples where every tuple contains both the word and its part-of-speech tag
        
    Parameters
    ----------
    sentence: str
         E.g. object-type column of a dataframe

    Returns
    -------
    list
       a list of tuples
    """

    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
         return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:                    
        return None

def lemmatize_sentence(sentence):
    """Take an entire sentence / string and outputs lemmatized sentence.

    There are some POS tags that correspond to words where the lemmatized form does not differ from the original word. 
        For these, [python]nltk2wn_tag()[/python] returns None and [python]lemmatize_sentence()[/python] 
        just copies them from the input to the output sentence.

    Parameters
    ----------
    sentence: str
         E.g. object-type column of a dataframe

    Returns
    -------
    list
       a list of strings
    """

    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))    
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:                        
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)

### Defining regex for preprocessing function

bad_symbols_re = re.compile('[^0-9a-z #+_]')
replace_by_space_re = re.compile('[-/(){}\[\]\|,;@=:]')

### Prepare list of punctuations to be excluded in cleaning process

punctuation = ['&', '%', '§', '/', '(', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '_', '*', '|', ',', '+', '€', '$', '#', '==', '[',']', '{', '}', "`", '´', "´´", '!', '°', '^', '--', "``", "''", '-', '"', "'", "?"]

### Write tailored preprocessing function to remove specific strings, symbols, code-snippets, stopwords and punctuation and to lemmatize words

def preprocessor(text_column):
    """Remove symbols, punctuation, code-snippets and stopwords and lemmatize text.

    Parameters
    ----------
    text_column : str
        E.g. object-type column of a dataframe

    Returns
    -------
    list
        a list of strings
    """

    ### Split text on capital letters
    text_column = ' '.join(re.findall('[a-zA-Z][^A-Z]*', text_column))
    ### Remove code snippets
    text_column = re.sub(r'<code>.*?</code>', ' ', text_column.lower())
    text_column = re.sub(r'<.*?>', ' ', text_column)
    text_column = re.sub(r'[( )]', ' ', text_column)
    ### Removing dots connecting words and at end of sentence
    text_column = " ".join(text_column.split("."))
    ### Removing backslash connecting words
    text_column = " ".join(text_column.split("\\"))
    ### Replace symbols defined in regex by a space
    text_column = replace_by_space_re.sub(' ', text_column)
    ### Delete symbols defined in regex from text
    text_column = bad_symbols_re.sub('', text_column)
    ### Remove stopwords from sentences
    text_column = [w for w in text_column.split() if w not in stopwords]
    text_column = ' '.join(text_column)
    ### Clean each letter/token in a sentence
    token = [letter for letter in text_column if letter not in punctuation]
    text_column = ''.join(token)
    ### Lemmatize sentences
    text_column = lemmatize_sentence(text_column)
    ### Replace occurrences of single alphabet characters
    text_column = re.sub(r"\b[a-zA-Z]\b", "", text_column)
    ### Remove unnecessary white spaces
    text_column = ' '.join(text_column.split())
    return text_column

example_sentence = "<p> ´´HI myself 3945p0hfnds <p> n, <code> grnkgljhkge </code>I am singing. 'our' (testing)   '  arbitrary & d lkögeaf_fewnefwl& can't c you're again tghghg.words.blub wild@fly\modules\system\\layers |tags|some more tags|blub"

preprocessor(example_sentence)

### Write tailored preprocessing function for tags

def tagcleaner(text_column):
    """Remove vertical lines from string and return only string.

        Parameters
        ----------
        text_column : str
            E.g. object-type column of a dataframe with vertical slash separators

        Returns
        -------
        list
            a list of strings
        """

    ### Removing vertical line connecting words
    text_column = text_column.split("|")
    text_column = ' '.join(text_column)
    return text_column

tag_example = "php|laravel|composer-php"

tagcleaner(tag_example)