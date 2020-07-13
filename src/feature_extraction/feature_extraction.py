### Import packages to create absolute file path & make code independent of operating system

from pathlib import Path
import os.path

import warnings
warnings.filterwarnings("ignore")

### Import packages for data manipulation

import pandas as pd
import numpy as np
import re

### Import packages to visualize data

import matplotlib.pyplot as plt
import seaborn as sns

### Import packages for feature extraction

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

### Import packages for modeling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
### from xgboost import XGBClassifier

### Import packages for model selection and performance assessment
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, log_loss, classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, mean_squared_error, f1_score


### Read in dataset
print(os.getcwd())

base_path = Path("__file__").parent
full_path = (base_path / "../../GitHub/FrauenLoop_NLP_Project_2020/data/processed/stackoverflow_preprocessed.csv").resolve()
# Depending on running this in interactive shell vs. terminal, I need to include GitHub/FrauenLoop_NLP_Project_2020 in filepath or not...
stackoverflow = pd.read_csv(os.path.join(full_path))

### Print out dataset for overview
print(stackoverflow.head())
print(stackoverflow.info())

### Define function to count number of words in an answer
def wordcounter(x):
    x = len(re.findall(r'\w+', str(x)))
    return x

### Feature extraction: Get wordcount of stackoverflow answers
stackoverflow['answer_wordcount'] = stackoverflow['answer_text_clean'].apply(wordcounter)

### Check of possible patterns in wordcount and answer score
stackoverflow.groupby(['score_cat', 'answer_wordcount']).size().unstack(fill_value=0)

### Make score_cat column into type category and assign numeric category codes
### Then make integer to be able to show heatmap in next step
stackoverflow['score_cat'] = stackoverflow['score_cat'].astype('category')
stackoverflow['score_cat'] = stackoverflow.score_cat.cat.codes
stackoverflow_copy = stackoverflow.copy()
stackoverflow_copy.head()
stackoverflow_copy['score_cat_int'] = stackoverflow_copy.score_cat.astype(int)

### Get heatmap to check for correlation of answer score and answer wordcount
df = pd.DataFrame(stackoverflow_copy, columns=['score_cat_int', 'answer_wordcount'])
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

### Creating a binary feature "code" holding info on whether or not stackoverflow answer contains code
def codecheck(x):
    x = 1 if '<code>' in x else 0
    return x

stackoverflow['code_binary'] = stackoverflow['answer_text'].apply(codecheck)
stackoverflow['code_binary'].value_counts()
stackoverflow.groupby(['score_cat', 'code_binary']).size().unstack(fill_value=0)

### Creating a feature "code count" holding info on how many code-snippets an answer contains
def codecount(x):
    x = x.count("<code>")
    return x

stackoverflow['code_count'] = stackoverflow['answer_text'].apply(codecount)
stackoverflow.groupby(['score_cat', 'code_count']).size().unstack(fill_value=0)

pd.set_option('display.max_colwidth', -1)

### Figure out how to deal with missing values for answer_text_clean, tags_clean and question_title_clean
print(stackoverflow.info())
stackoverflow.isna().any()
### decide if I want to drop missing values? stackoverflow.dropna()

### Split into predictors and outcome data
y = stackoverflow['score_cat']
X = stackoverflow.drop(['score_cat', 'score', 'answer_count', 'comment_count', 'creation_date', 'favorite_count', 'view_count'] , axis=1)  

y.dtype

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
                                     ngram_range = (1, 1), max_features = 100) 
                                     # if I remove max_features, I get error that array shape X and Y do not align

        ### Fit to data
        X_train = vectorizer.fit_transform(df[name[0]].values.astype(str))
        ## X_train = X_train.toarray()
        # is this needed? how do I address mismatching shape problem

        ### Return sparse matrix
        return X_train
    
    def fit(self, df, y=None):
        ### Unless error returns self
        return self

### Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

print(y_test)

### Model selection process: Create list of different classifiers/algorithms to try out

classifiers = [
    KNeighborsClassifier(),
    RandomForestClassifier(random_state=1),
    GradientBoostingClassifier(random_state=1)
    ## XGBClassifier(random_state=1)
    ]

### Model selection process: Loop through the different classifiers using the pipeline

for classifier in classifiers:
    model_pipeline = Pipeline([
        ('feats', FeatureUnion([
             # Ngrams
            ('ngram_all', Ngrams(X_train[['answer_text_clean']]))])),
            # Classifier
            ('classifier', classifier)])
    model_pipeline.fit(X_train, y_train)
    y_predict = model_pipeline.predict(X_test)
    print(classifier)
    print(y_predict)
    print("model score: %.3f" % model_pipeline.score(y_predict, y_test)


## confusion_matrix(y_test, grid_search.predict(X_test))

  # confm_hold = confusion_matrix(y_test, y_predict)
    # print(confm_hold)

# np.array(s)
## confm_hold_df = pd.DataFrame(confm_hold, index = ['No Medal', 'Medal'],
                               # columns = ['No Medal', 'Medal'])
## plt.figure(figsize=(5,4))
## sns.heatmap(confm_hold_df, annot=True, fmt=".4f", linewidths=.5, square = True)
