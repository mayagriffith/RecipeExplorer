#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:05:27 2024

@author: Alaina

Evaluating Model Performance
"""

import user_pref as pref
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score, f1_score
from matplotlib import pyplot as plt
#%% User Preference Model User Independent Evaluation
"""
User-independent evaluation of user preference model. Includes classification 
report, ROC curve with AUC, confusion matrix. Shows that our model is slightly 
better than random, but may not be the best approach due to limited data 
(we have only 5 subjects). See below for LOGOCV (leave one group out CV)
"""
# get data- compile all user data for user- independent evaluation
path = 'cleanedVotes/'
names = ["Alaina", "maya", "nate", "nick", "Rider"] 
# initialize df for combined votes
combined_votes = pd.DataFrame()
for name in names:
    file_path = f"{path}{name}.csv"
    temp_df = pd.read_csv(file_path, header=None, names=["title", "vote"])
    # add column for name so we have user IDs
    temp_df["user_id"] = name
    combined_votes = pd.concat([combined_votes, temp_df], ignore_index=True)
    

cleaned_recipes_df = pd.read_csv('RecipeData/cleaned_recipes.csv')


combined_df = pd.merge(cleaned_recipes_df, combined_votes, on='title', how='inner')

# split dataset into train and test based on user id for user independent evaluation
# using 20% of data for testing
train_users, test_users = train_test_split(names, test_size=.2, random_state=42)
train_combined = combined_df[combined_df["user_id"].isin(train_users)]
test_combined = combined_df[combined_df["user_id"].isin(test_users)]

# drop user id for feature selection
train_combined = train_combined.drop(["user_id"], axis=1)
test_combined = test_combined.drop(["user_id"], axis=1)

# find the most important features
selector = VarianceThreshold(threshold=0)
selector.fit(train_combined.drop(['title', 'vote'], axis=1))

# get the features that have some variance
features_with_variance = train_combined.drop(['title', 'vote'], axis=1).columns[selector.get_support()]

# Preparing the data
X_train = train_combined[features_with_variance]
y_train = train_combined['vote']

X_test = test_combined[features_with_variance]
y_test = test_combined['vote']

# train model by calling user_pref
log_reg_model, y_pred, y_scores = pref.train_predict_model(X_train, y_train, X_test)

# evaluate preformance
# Classification report
report = classification_report(y_test, y_pred)
print(report)
# ROC-AUC- binary classification so we can use sklearns fntns
# below gives false pos rate, true pos rate, thresholds for decision fntn
fpr, tpr, thresholds = roc_curve(y_test, y_scores) 
# area under curve
roc_auc = auc(fpr, tpr)

# plot
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for User Preference Model with User-Independent Evaluation")
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("User Preference User Independent Confusion Matrix\n", conf_matrix)
plt.figure()
plt.imshow(conf_matrix)
plt.colorbar()
plt.xlabel("True Class")
plt.ylabel("Predicted Class")
plt.xticks(ticks=range(len(conf_matrix)), labels=['Class 0 (Dislike)', 'Class 1 (Like)'])
plt.yticks(ticks=range(len(conf_matrix)), labels=['Class 0 (Dislike)', 'Class 1 (Like)'])
plt.title("Confusion Matrix Heatmap for User Preference Model with User-Independent Evaluation")
plt.tight_layout()

#%% User Preference Model LOGOCV Evaluation
"""
LOGOCV (leave one group out cross validation) evaluation of user preference 
model. 
"""
# vote is target, user_id is user id (group), features are other col values
# use combined_df for data
# drop title from combined_df
CV_df = combined_df.drop(["title"], axis=1)
# Features
X = CV_df.drop(columns=["user_id","vote"]).values
# target
y = CV_df["vote"].values

# define groups
groups = CV_df["user_id"].values

# initialize logo
logo = LeaveOneGroupOut()

# initialize lists to hold accuracy scores and f1 scores
# accuracy is being used because we have a balanced dataset- each user
# submitted 50 likes and 50 dislikes
# will compute f1 as well for more robust evaluation
accuracy_CV = []
f1_CV = []

# initialize lists to store true labels, predicted scores, and outcomes
# to later create ROC curve and conf. matrix
true_labels = []
pred_scores = []
pred_labels = []

# preform LOGOCV
for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # feature selection
    selector = VarianceThreshold(threshold=0)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)
    
    # initialize and train model
    log_reg_model, y_pred, y_scores = pref.train_predict_model(X_train, y_train, X_test)
    
    # evaluate
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_CV.append(accuracy)
    f1 = f1_score(y_test, y_pred, average="binary")
    f1_CV.append(f1)
    
    # store true and pred labels and y/ pred scores in flat lists
    true_labels.extend(y_test)
    pred_labels.extend(y_pred)
    pred_scores.extend(y_scores)
    
# get average accuracy
avg_acc = sum(accuracy_CV)/len(accuracy_CV)
print("LOGOCV Average Accuracy: ", avg_acc)
# get average f1
avg_f1 = np.mean(f1_CV)
print("LOGOCV Average F1: ", avg_f1)

# compute and plot confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)
print("User Pref LOGOCV Confusion Matrix\n", conf_matrix)
plt.figure()
plt.imshow(conf_matrix)
plt.colorbar()
plt.xlabel("True Class")
plt.ylabel("Predicted Class")
plt.xticks(ticks=range(len(conf_matrix)), labels=['Class 0 (Dislike)', 'Class 1 (Like)'])
plt.yticks(ticks=range(len(conf_matrix)), labels=['Class 0 (Dislike)', 'Class 1 (Like)'])
plt.title("Confusion Matrix Heatmap for User Preference Model with LOGOCV")
plt.tight_layout()

# compute and plot ROC curve with AUC
fpr, tpr, thresholds = roc_curve(true_labels, pred_scores) 
roc_auc = auc(fpr, tpr)
roc_auc = round(roc_auc, 4)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for User Preference Model with LOGOCV")
plt.legend(loc="lower right")
plt.show()

"""
As expected, LOGOCV shows slightly worse model preformance as it is a more
robust technique for evaluating a model when data is limited like we have
in this case. 
"""
#%% Train vs test error for user preference model
"""
Easy way to tell if our model is over or underfitting to have more indication
of next steps
"""
CV_df = combined_df.drop(["title"], axis=1)
# Features
X = CV_df.drop(columns=["user_id","vote"]).values
# target
y = CV_df["vote"].values

# split
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# store train sizes- 9 evenly spaced nums from .1 to .9 (fractions of train data to use)
train_sizes = np.linspace(.1, .99, 9)

# initialize variables to store train and test error
train_error_all = []
test_error_all = []

for size in train_sizes:
    # smaller subset split
    X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all, train_size=size, random_state=42)
    
    # train model
    log_reg_model, y_pred, y_scores = pref.train_predict_model(X_train, y_train, X_test)
    
    # get and store errors
    # using 1-accuracy because classification with balanced dataset
    train_error = 1 - accuracy_score(y_train, log_reg_model.predict(X_train))
    train_error_all.append(train_error)
    
    test_error = 1 - accuracy_score(y_test, y_pred)
    test_error_all.append(test_error)
    
# plot
plt.figure()
plt.plot(train_sizes*100, train_error_all, label="Train")
plt.plot(train_sizes*100, test_error_all, label="Test")
plt.xlabel("Training Data Used")
plt.ylabel("Error (1-Accuracy)")
plt.title("Train vs. Test Error For User Preference Model")
plt.legend()
plt.show()

"""
The plot shows relatively low training error with relatively high test error,
representing that our model may be overfitting.
"""

"""
NEXT STEPS
Overall, this model is a good starting point but there is certainly room for 
improvement as it is only slightly better than random. Next steps: investigate
ways to improve feature selection, use random search to fine tune model 
parameters, implement regularization, attempt to get more data.
"""