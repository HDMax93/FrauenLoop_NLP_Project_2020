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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

### Import packages for model selection, tuning and performance assessment
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn import metrics
from sklearn.metrics import accuracy_score, log_loss, classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, mean_squared_error, f1_score
from sklearn.decomposition import PCA, NMF
from sklearn.decomposition import TruncatedSVD
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll import scope

### Import packages to visualize data
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

### Saving the final model
import pickle

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
    LinearDiscriminantAnalysis(),
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
            # No. of code snippets
            ('codecounter', CodeCounter(X_train)),
            # JaccardSimilarity of question and answer
            ('jaccard', JaccardSimilarity(X_train)),
            # Similarity of question and answer
            ('similarity', Similarity(X_train)),
            # Top tags present
            ('toptags', TopTagEncoder(X_train))
            ])),
            # PCA
            ('reduce_dim', TruncatedSVD(n_components=150)),
            # Classifier
            ('classifier', GradientBoostingClassifier(random_state=1)
        )
                        ])


model_pipeline.fit(X_train, y_train)
y_predict = model_pipeline.predict(X_test)
print("GradientBoostingClassifier")
print(metrics.classification_report(y_test, y_predict))


### Define the model cross-validation configuration

cv = KFold(n_splits=5, shuffle=True, random_state=1)
# cross_val_score(model_pipeline, X_train, y_train, cv=cv)

param_hyperopt = {
    'reduce_dim': hp.choice('reduce_dim', [TruncatedSVD()]),
    'reduce_dim__n_components': scope.int(hp.quniform('n_components', 150, 300, 50)),
    'classifier__n_estimators': scope.int(hp.quniform('n_estimators', 1250, 2500, 500)),
    'classifier__max_features': hp.choice('max_features', ['auto', 'sqrt']),
    'classifier__max_depth': scope.int(hp.quniform('max_depth', 25, 50, 1)),
    'classifier__min_samples_split': scope.int(hp.quniform('min_samples_split', 5, 10, 2)),
    'classifier__min_samples_leaf': scope.int(hp.quniform('min_samples_lead', 2, 4, 1)),
}

print(param_hyperopt)

def objective(params):
    model_pipeline.set_params(**params)
    shuffle = KFold(n_splits=5, shuffle=True, random_state=1)
    score = cross_val_score(model_pipeline, X_train, y_train, cv=shuffle, n_jobs=1)
    # binarize to have ROC-scoring
    return 1-score.mean()

### The trials object will store details of each iteration
trials = Trials()

### Run the hyperparameter search using the tpe algorithm
best = fmin(fn = objective,
            space = param_hyperopt,
            algo = tpe.suggest,
            max_evals = 10,
            trials = trials,
            rstate= np.random.RandomState(1))


### Get the values of the optimal parameters
best_params = space_eval(space, best)

### Fit the model with the optimal hyperparamters
model_pipeline.set_params(**best_params)
model_pipeline.fit(X_train, y_train)

### Score with the test data
y_score = model_pipeline.predict_proba(X_test)
# auc_score = roc_auc_score(y_test, y_score[:,1])
y_predict = model_pipeline.predict(X_test)
print("GradientBoostingClassifier")
print(metrics.classification_report(y_test, y_predict))

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
               'classifier__min_samples_split': [2],
               'classifier__min_samples_leaf': [2]}

print(param_grid)

### Choose best-performing parameters using GridSearchCV

grid_classifier = GridSearchCV(model_pipeline, param_grid = param_grid, cv=cv, iid=False, n_jobs=-1, refit = True)
# scoring='roc_auc' --> reincorporate
grid_classifier.fit(X_train, y_train)
print("Best result: %f using parameters %s" % (grid_classifier.best_score_, grid_classifier.best_params_))


### Assess model performance on test data

print("Model Score assessed on test data: %.3f" % grid_classifier.score(X_test, y_test))
print("Classification Report:", classification_report(y_test, grid_classifier.predict(X_test)))

### Visualize the confusion matrix

lst    = [grid_classifier]
length = len(lst)
mods   = ['GradientBoostingClassifier']
fig = plt.figure(figsize=(15,12))
fig.set_facecolor("#F3F3F3")
for i,j,k in itertools.zip_longest(lst,range(length),mods) :
    plt.subplot(1,1,j+1)
    conf_matrix = confusion_matrix(y_test, i.predict(X_test))
    sns.heatmap(conf_matrix,annot=True,fmt = "d",square = True,
                xticklabels=["predicted bad","predicted good", "predicted great"],
                yticklabels=["actual bad","actual good", "actual great"],
                linewidths = 2,linecolor = "w",cmap = "RdYlGn")
    plt.title(k,color = "g")
    plt.subplots_adjust(wspace = .3,hspace = .3)


### Function to plot confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


cnf_matrix = confusion_matrix(y_train, y_test,labels=['bad', 'good', 'great'])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['bad', 'good', 'great'],
                      title='Gradient Boosting Classifier - Confusion Matrix')

### Plot the AUC/ROC-curve

lst    = [grid_classifier]
length = len(lst)
mods   = ['GradientBoostingClassifier']
fig = plt.figure(figsize=(10,10))
fig.set_facecolor("#F3F3F3")
probabilities = grid_classifier.predict_proba(X_test)
predictions   = grid_classifier.predict(X_test)
fpr,tpr,thresholds = roc_curve(y_test,probabilities[:,1])
plt.plot(fpr,tpr,linestyle = "dotted",
            color = "royalblue",linewidth = 2,
            label = "AUC = " + str(np.around(roc_auc_score(y_test,predictions),3)))
plt.plot([0,1],[0,1],linestyle = "dashed",
        color = "orangered",linewidth = 1.5)
plt.legend(loc = "lower right",
        prop = {"size" : 14})
plt.grid(True,alpha = .15)
plt.title(k,color = "b")
plt.xticks(np.arange(0,1,.3))
plt.yticks(np.arange(0,1,.3))

### Save the model to disk
base_path = Path("__file__").parent
full_path = (base_path / "../models/finalized_model.sav").resolve()
filename = full_path
outfile = open(filename, 'wb')
pickle.dump(grid_classifier, outfile)
outfile.close()

### Load the model from disk

infile = open(filename,'rb')
loaded_model = pickle.load(infile)
infile.close()
result = loaded_model.score(X_test, y_test)
print(result)