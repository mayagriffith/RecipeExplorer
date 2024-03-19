#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:05:27 2024

@author: Alaina

Evaluating Model Performance
"""
#%% User Preference Model
import user_pref as pref
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from matplotlib import pyplot as plt

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
print(roc_auc)
# plot
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for User Preference Model")
plt.legend(loc="lower right")
plt.show()
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
plt.figure()
plt.imshow(conf_matrix)
plt.colorbar()
plt.xlabel("True Class")
plt.ylabel("Predicted Class")
plt.title("Confusion Matrix Heatmap for User Preference Model")