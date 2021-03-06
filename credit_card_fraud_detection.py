# -*- coding: utf-8 -*-
"""Credit-card-fraud-detection-P1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MVDs8-tqv_-yvnzElJZ0uBKw7S8dRCIy

Importing the Dependencies
"""

import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset to a Pandas DataFrame
# load the dataset from https://www.kaggle.com/mlg-ulb/creditcardfraud 
zf = zipfile.ZipFile("C:/Users/Rodolphe/Documents/COURS/2A/AADA/Projet_apprentissage_dynamique/creditcard.csv.zip")
credit_card_data = pd.read_csv(zf.open("creditcard.csv"))

# display the first 5 rows of the dataset
credit_card_data.head()

# display the last 5 rows of the dataset
credit_card_data.tail()

# display the information about the dataset
credit_card_data.info()

# see if you have features with missing data
credit_card_data.isnull().sum()

# distribution of legit and fraudulent transactions we can see that the data is higly imbalanced
credit_card_data["Class"].value_counts()

"""0 --> normal transaction
1 --> fraudulent transaction
"""

# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)

# statistical measures of the data
legit.Amount.describe()

# the mean amount of fraudluent transactions is much bigger than the legit transaction
fraud.Amount.describe()

# compare the values for both transactions
credit_card_data.groupby('Class').mean()

"""Under sampling in order to build dataset of similar distribution of normal and fraud transactions """

legit_sample = legit.sample(n=492)

# concatenating two DataFrames
new_dataset = pd.concat([legit_sample, fraud], axis = 0)

new_dataset['Class'].value_counts()

# look to the statistics of the new balanced dataset and compare them with the ones of the imbalanced dataset to see that the statistics still coherent to be sure that we have a good sample
new_dataset.groupby(['Class']).mean()

# splitting the data into features and targets
X = new_dataset.drop(columns = 'Class', axis = 1)
Y = new_dataset['Class']

# split the data into Training and testing data sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify = Y, random_state = 2) # stratify to balance the  data according to their class in the test and training sets

# Model training
# logistic regression
model = LogisticRegression()

# training the LR with X_train
model.fit(X_train, Y_train)

# model evaluation
# Accuracy score
X_train_pred = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_pred, Y_train)
print('Accuracy on training data', training_data_accuracy)
X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, Y_test)
print('Accuracy on test data set', test_data_accuracy)

"""Accuracy score = (nombre de transactions legit class??es comme l??git + nombre de trasnactions de fraud class??es comme fraud)/nb total des transactions class??es 
= 90%
10% d'erreur
nombre de transactions legit class??es comme fraud + nombre de transactions de fraud clas??e comme l??git

accuracy test is close to training accuracy which means our model is properly fitted. If accuracy test is much higher than the training accuracy score it means the model is underfitted
"""

