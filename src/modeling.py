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

### Import packages for feature extraction from 
from common_utils.feature_helpers import JaccardSimilarity, Similarity, WordCounter, CodeCheck, CodeCounter, Ngrams, TopTagEncoder

### Import packages for modeling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

### Import packages for model selection and performance assessment
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn import metrics
from sklearn.metrics import accuracy_score, log_loss, classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, mean_squared_error, f1_score

### Read in dataset

print(os.getcwd())

base_path = Path("__file__").parent
full_path = (base_path / "../data/processed/stackoverflow_modeling.csv").resolve()
stackoverflow = pd.read_csv(os.path.join(full_path))

stackoverflow.info()

### Split into predictors and outcome data

y = stackoverflow['score_cat_all']
X = stackoverflow.drop(['answer_score', 'question_score', 'answer_count', 'comment_count', 'creation_date', 'view_count', 'score_cat_all'] , axis=1)


### Split into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


### Model selection process: Create list of different classifiers/algorithms to try out

classifiers = [
    LogisticRegression(random_state=1),
    KNeighborsClassifier(),
    SVC(random_state=1),
    DecisionTreeClassifier(random_state=1),
    RandomForestClassifier(random_state=1),
    GradientBoostingClassifier(random_state=1)
    ]

### Model selection process: Loop through the different classifiers using the pipeline

for classifier in classifiers:
    model_pipeline = Pipeline([
        ('feats', FeatureUnion([
            # Ngrams
            ('ngram', Ngrams(X_train[['answer_text_clean']])),
            # Wordcounter
            ('wordcount', WordCounter(X_train[['answer_text_clean']])),
            # Code contained
            ('codecheck', CodeCheck(X_train)),
            # No. of code snippets
            ('codecounter', CodeCounter(X_train)),
            # JaccardSimilarity of question and answer
            ('jaccard', JaccardSimilarity(X_train)),
            # Similarity of question and answer
            ('similarity', Similarity(X_train)),
            # Top tags present
            ('toptags', TopTagEncoder(X_train))
            ])),
            # Classifier
            ('classifier', classifier)])
    model_pipeline.fit(X_train, y_train)
    y_predict = model_pipeline.predict(X_test)
    print(classifier)
    print(metrics.classification_report(y_test, y_predict))

### Setting up model pipeline with best-performing classifier

model_pipeline = Pipeline([
    ('feats', FeatureUnion([
            # Ngrams
            ('ngram', Ngrams(X_train[['answer_text_clean']])),
            # Wordcounter
            ('wordcount', WordCounter(X_train[['answer_text_clean']])),
            # Code contained
            ('codecheck', CodeCheck(X_train)),
            # No. of code snippets
            ('codecounter', CodeCounter(X_train)),
            # JaccardSimilarity of question and answer
            ('jaccard', JaccardSimilarity(X_train)),
            # Similarity of question and answer
            ('similarity', Similarity(X_train)),
            # Top tags present
            ('toptags', TopTagEncoder(X_train))
            ])),
            # Classifier
            ('classifier', GradientBoostingClassifier(random_state=1))])
model_pipeline.fit(X_train, y_train)
y_predict = model_pipeline.predict(X_test)
print("GradientBoostingClassifier")
print(metrics.classification_report(y_test, y_predict))


### Define the model cross-validation configuration

cv = KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(model_pipeline, X_train, y_train, cv=cv)


### Choose best-performing model to tune using random hyperparameter grid

# Number of boosting stages to perform
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider when looking for the best split
max_features = ['auto', 'sqrt']
# Maximum depth of the individual regression estimators (limits the number of nodes in the tree)
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split an internal node
min_samples_split = [2, 5, 10]
# Minimum number of samples required to be at a leaf node
min_samples_leaf = [1, 2, 4]

### Create random grid
random_grid = {'classifier__n_estimators': n_estimators,
               'classifier__max_features': max_features,
               'classifier__max_depth': max_depth,
               'classifier__min_samples_split': min_samples_split,
               'classifier__min_samples_leaf': min_samples_leaf}

print(random_grid)


### Find best combination of parameters using randomized hyperparameter search

random_grid_classifier = RandomizedSearchCV(model_pipeline, param_distributions = random_grid, n_iter = 100, cv = cv, verbose=2, random_state=42, n_jobs = -1)
random_grid_classifier.fit(X_train, y_train)
print(random_grid_classifier.best_params_)
print(random_grid_classifier.best_score_)

### Create param grid based on results from random grid search

param_grid = {'classifier__n_estimators': [3000],
               'classifier__max_features': ['sqrt'],
               'classifier__max_depth': [None],
               'classifier__min_samples_split': [1, 2],
               'classifier__min_samples_leaf': [1, 2]}

print(param_grid)

### Choose best-performing parameters using GridSearchCV

grid_classifier = GridSearchCV(model_pipeline, param_grid = param_grid, cv=cv, iid=False, n_jobs=-1, refit = True)
# scoring='roc_auc' --> reincorporate
grid_classifier.fit(X_train, y_train)
print("Best result: %f using parameters %s" % (grid_classifier.best_score_, grid_classifier.best_params_))


### Assess model performance on test data

print("Model Score assessed on test data: %.3f" % grid_classifier.score(X_test, y_test))
print("Classification Report:", classification_report(y_test, grid_classifier.predict(X_test)))
