### Import packages for preprocessing

import pandas as pd
from utils.nltk_helpers import nltk2wn_tag, lemmatize_sentence
from utils.nltk_helpers import preprocessor, tagcleaner

### Import packages to create absolute file path & make code independent of operating system

from pathlib import Path
import os.path
import warnings
warnings.filterwarnings("ignore")

### Import packages to visualize data

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

### Read in dataset

base_path = Path("__file__").parent

full_path = (base_path / "../../data/raw/stackoverflow_raw_finalsample_all.csv").resolve()

stackoverflow = pd.read_csv(os.path.join(full_path))

stackoverflow.shape

stackoverflow.head()


### Drop any columns from dataframe not holding useful information

stackoverflow = stackoverflow.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'question_id', 'question_id_check'])

### Print out variable types for overview

stackoverflow.info()

### Figure out how to deal with missing values for answer_text_clean, tags_clean and question_title_clean

stackoverflow.isna().any()

#### Histogram of distribution of outcome column "score"

f, ax = plt.subplots(figsize=(20,20))
sns.countplot(x='answer_score', data=stackoverflow)
plt.xlim(None, 100) 
plt.ylim(0, 9000) 

base_path = Path("__file__").parent
full_path = (base_path / "../../reports/figures/stackoverflow_answerscore_distribution_sampled.png").resolve()
plt.savefig(full_path, dpi=300)

plt.show()


### Separating target and text columns

excluded_cols = ['comment_count', 'creation_date', 'view_count', 'answer_count', 'score_cat_positive', 'question_score', 'answer_score']
target_col = ['score_cat_all']
text_cols = [x for x in stackoverflow if x not in target_col + excluded_cols]
print(text_cols)


### Applying cleaning / preprocessing to all text columns

for var_name in text_cols:
    new_var = "%s_%s" % (var_name, "clean")
    stackoverflow[new_var] = stackoverflow[var_name].apply(preprocessor)


### Applying tag cleaner function to tags column

stackoverflow['tag_list_clean'] = stackoverflow['tags'].apply(tagcleaner)


### Check if preprocessed text looks as desired

pd.set_option('display.max_colwidth', -1)
stackoverflow.head(20)

### Check again for missing values

stackoverflow.info()

# Make categorical score labels numeric

stackoverflow['score_cat_all'] = stackoverflow['score_cat_all'].astype('category')
stackoverflow['score_cat_all'] = stackoverflow.score_cat_all.cat.codes


### Save preprocessed data to a csv file

base_path = Path("__file__").parent
full_path = (base_path / "../../data/processed/stackoverflow_preprocessed_all.csv").resolve()
stackoverflow.to_csv(os.path.join(full_path))