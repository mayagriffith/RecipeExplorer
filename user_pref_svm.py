#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:20:44 2024

@author: Alaina Birney (adapted from Maya's logistic regression model)

Using SVM for user preference model
"""
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

hyperparams = { 'kernel': 'sigmoid', 'C': 2.1544346900318843, "probability":True, "gamma": .001 }
# {'C': 1.6378937069540647, 'gamma': 0.0022758459260747888, 'kernel': 'sigmoid', 'roc_auc': 0.6106605188600633}

def train_predict_model(X_train, y_train, X_test, **kwargs):
    # train SVM
    svm_model = SVC(max_iter=10000, **kwargs)
    svm_model.fit(X_train, y_train)
    
    # get y scores for ROC curve (probailities for positive class)
    y_scores = svm_model.predict_proba(X_test)[:,1]
    
    # predicting on the test set
    y_pred = svm_model.predict(X_test)
    return svm_model, y_pred, y_scores

def standardize(X_train, X_test, is_kfold=False):
    # standardize features
    scaler = StandardScaler()
    # fit to train to avoid leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if is_kfold == False:
        # convert the scaled arrays back to DataFrames and preserve col names
        # only do this if not doing kfold, this is needed for the plot of feature 
        # importance, which isn't created when kfold is implemented 
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        return X_train, X_test
    else:
        return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    # get user name to access votes
    user_name = input("Please enter your username\n ")
    user_path = 'cleanedVotes/'+ user_name + '.csv'
    user_votes_df = pd.read_csv(user_path, header=None, names=['title', 'vote'])
    cleaned_recipes_df = pd.read_csv('RecipeData/cleaned_recipes.csv')
    
    combined_df = pd.merge(cleaned_recipes_df, user_votes_df, on='title', how='inner')
    
    # find the important features
    selector = VarianceThreshold(threshold=0)
    selector.fit(combined_df.drop(['title', 'vote'], axis=1))
    
    # get the features that have some variance
    features_with_variance = combined_df.drop(['title', 'vote'], axis=1).columns[selector.get_support()]
    
    # Preparing the data
    X = combined_df[features_with_variance]
    y = combined_df['vote']
    
    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train, X_test = standardize(X_train, X_test)
    
    svm_model, y_pred, y_scores = train_predict_model(
        X_train, y_train, X_test, **hyperparams)
    
    
    # Identifying recipes not seen before (not voted on)
    all_recipes = set(cleaned_recipes_df['title'])
    voted_recipes = set(user_votes_df['title'])
    unvoted_recipes = all_recipes - voted_recipes
    
    unvoted_recipes_df_corrected = cleaned_recipes_df[cleaned_recipes_df['title'].isin(unvoted_recipes)]
    
    # Predicting Useer's preferences on these recipes
    predicted_likes_corrected = svm_model.predict(unvoted_recipes_df_corrected[features_with_variance])
    
    # extracting titles of recipes predicted as likes
    recommended_recipes_titles = unvoted_recipes_df_corrected.loc[predicted_likes_corrected == 1, 'title']
    
    # displaying a few recommended recipes
    print("Recommended Recipes")
    print(recommended_recipes_titles.head())

    print("Calculating feature importance...")
    # find the most important features for the model using permutation feature importance b/c nonlinear kernel
    result = permutation_importance(svm_model, X_train, y_train, n_repeats=10, random_state=42)
    
    # get indices of features sorted by importance
    # values are indices of X_train, index 0 of sorted_idx corresponds to least important feature
    sorted_idx = result.importances_mean.argsort()
    # now index 0 of sorted_idx corresponds to most important feature
    sorted_idx = sorted_idx[::-1]

    # get top 10 scores and names
    top_10_feat = sorted_idx[:10]
    top_10_importances = result.importances_mean[top_10_feat]
    top_10_feature_names = X_train.columns[top_10_feat]
    
    # magnitude, not directionality is represented in this plot
    plt.figure()
    plt.bar(top_10_feature_names, top_10_importances)
    plt.title('Top 10 Features Influencing User\'s Preferences (directionality not represented)')
    plt.ylabel('Level of Influence')
    plt.xlabel('Feature')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # print accurracy and f1
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy ", accuracy)
    
    f1 = f1_score(y_test, y_pred)
    print("F1: ", f1)
    
