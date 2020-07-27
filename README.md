NOTE: This is work-in-progress. Collaborative NLP project as part of FrauenLoop ML course in summer 2020

## Predicting the Helpfulness of Stackoverflow Answers

This project was part of a Machine Learning course hosted by [FrauenLoop](https://www.frauenloop.org/).


### Project & Objective

*What if Stack Overflow provided its users with a prediction of how helpful the user's answer is likely going to be to the community?*

Most aspiring developers, analysts, and data scientists depend on and owe much of their learning to **online fora** where experienced coders can answer questions of more junior professionals and learners. According to Stack Overflow’s annual Developer Survey 2019, each month, about **50 million people visit Stack Overflow** to learn, share, and build their careers. Stack Overflow, consequently, is a powerful tool to expand knowledge and self-improve. By extension, the **quality of the answers** provided by the Stack Overflow community makes a big difference to **users' progress**. 

The **purpose** of this project was to **develop a model** that would provide users on Stack Overflow with an **advance estimate** of how helpful their answer to a question is going to be to other users. Past questions, answers and answer scores were used to train a model that would predict if an answer was likely to receive a **bad**, **good** or **great** score. 

The project objectives were to:

- Empower users to edit and improve their answers before publishing them through live predictions, to maximize their usefulness to the community.
- Improve the understanding of what factors make a high quality answer, to educate users on being as helpful as possible to the community.

Such a live prediction/rating of answers can be of use across online help fora, beyond Stack Overflow.

### Key Insights

1. Using GradientBoostingClassifier, an accuracy of 0.51 was reached, leaving much room for improvement.
2. Contrary to intial assumptions, the text data consisting of answers, questions and answer tags could not sufficiently help differentiate between a bad, good and great answer.
3. Additional data and features are needed to provide predictions on the quality of user answers with a high accuracy.

### Data

- The data for this project was retrieved using Goolge BigQuery API. Based on a dataset containing all Stack Overflow questions and answers between May 2019 and May 2020, an equal amount of bad (answer score < 0), good (answer score of 1-6) and great answers (answer score of 7+) were retrieved, resulting in a dataset with 30,0000 observations.
- You can learn more about the dataset and explore it using SQL [here](https://console.cloud.google.com/marketplace/product/stack-exchange/stack-overflow?project=frauenloop-nlp-2020&folder=&organizationId=). 

### Navigating this repo

- [Requirement](PREDICTING-HELPFULNESS-OF-STACKOVERFLOW-ANSWERS): Before starting your project, make sure to set up a project environment and to install all the required packages for the project. 
- You will find the relevant code to execute in the [source folder](src/)
    - [data retrieval folder](src/api_data_retrieval): If you would like to go through the process of retrieving the original dataset yourself, you will find the code in this folder. You will also need to follow these steps to set up a Google BigQuery:
        1. Set up an account on [Google Cloud Platform](https://console.cloud.google.com/).
        2. Follow [steps 1-4](https://cloud.google.com/bigquery/docs/quickstarts/quickstart-client-libraries#client-libraries-install-python) on using client libraries.
        3. Place the JSON file (see 2) with your key in your directory under "PREDICTING-HELPFULNESS-OF-STACKOVERFLOW-ANSWERS/data/raw/GoogleBigQuery_key.json" and name it "GoogleBigQuery_key.json".
    **Alternatively**, just download the "final sample" dataset [here](https://drive.google.com/file/d/1ve6gzOKgJhdESAv2MLImbqZRi4VsSL5q/view?usp=sharing) and place it [here](data/raw) in your repository. If you use this dataset, you will not need to execute the data_retrieval_and_sampling.py file, start by directly running the preprocessing.py script instead.
    - [data exploration](src/data_exploration): You will find some code to explore the data visually in this folder. Graphs are saved in the [reports/figures](reports/figures) folder.

    ![Distribution of Answer Scores](https://raw.githubusercontent.com/HDMax93/Predicting-Helpfulness-Of-Stackoverflow-Answers/master/reports/figures/stackoverflow_answerscore_distribution.png)

    - [data manipulation](src/data_manipulation): Run this script to preprocess the data you retrieved using Google BigQuery API or that you downloaded.
    - [feature extraction](src/feature_extraction): This folder contains the script for testing if the features are extracted as desired, using the [feature extraction classes](src/common_utils).
    - [model training](src/model_training): You will find the script for choosing, training and hypertuning the model here.
- [final model](models): If you execute the above scripts, your final model will be stored in this place. If you would like to download the final model directly, you can do so by clicking on [this link](DOWNLOAD LINK HERE).
- [notebooks](notebooks): This folder contains some exploratory notebooks created in the process of finding the model.

### Giving feedback

Did you spot any mistakes or would like to provide feedback on improving the model? Please [reach out to me](mailto:henriekemax@googlemail.com)?subject=[Feedback-On-GitHub-Stackoverflow-Project], I would love to hear and learn from you!

### Acknowledgements 

XXXXXX