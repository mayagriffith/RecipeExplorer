#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:20:44 2024

@author: Alaina Birney (adapted from Maya Griffith's logistic regression model)

Using SVM for user preference model
"""
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

hyperparams = { 'kernel': 'sigmoid', 'C': 1778.2794, "probability":True, "gamma": 0.0001 } # best so far
"""
getting pretty good results for the amount of training data provided
"""
#hyperparams = { 'kernel': 'rbf', 'C': 1000, "probability":True, "gamma": 0.031 } # also very good
#hyperparams = { 'kernel': 'sigmoid', 'C': 100, "probability":True, "gamma": 0.001 } # works the best out of the ones so far!
#hyperparams = { 'kernel': 'sigmoid', 'C': 2.1544346900318843, "probability":True, "gamma": .001 }
#hyperparams = { 'kernel': 'sigmoid', 'C': 1.6378937069540647, "probability":True, "gamma": 0.0022758459260747888 }
# {'C': 1.6378937069540647, 'gamma': 0.0022758459260747888, 'kernel': 'sigmoid', 'roc_auc': 0.6106605188600633}

def train_predict_model(X_train, y_train, X_test, **kwargs):
    # train SVM
    svm_model = SVC(**kwargs) # removed max iters
    svm_model.fit(X_train, y_train)
    
    # get y scores for ROC curve (probailities for positive class)
    y_scores = svm_model.predict_proba(X_test)[:,1]
    
    # predicting on the test set
    y_pred = svm_model.predict(X_test)
    return svm_model, y_pred, y_scores

def standardize_and_reduce(X_train, X_test, y_train, importance_threshold,
                           unvoted_recipes=None, prediction_data=None, is_kfold=False):
    """
    Standardize non-binary features and reduce feature space with PCA
    """
    
    non_b_cols = ["rating", "calories", "protein", "fat", "sodium"]
    
    preprocessor = ColumnTransformer(transformers = [("num", StandardScaler(), non_b_cols)], remainder="passthrough")
    # standardize non-binary features and convert back to df
    # fit to train to avoid leakage
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    """
    # drop nonb cols from X_train, add nonb_scaled
    X_train.drop(labels = non_b_cols, axis=1, inplace=True)
    X_test.drop(labels = non_b_cols, axis=1, inplace=True)
    
    X_train = pd.concat([X_train, X_train_nonb_scaled], axis=1)
    X_test = pd.concat([X_test, X_test_nonb_scaled], axis=1)

    # drop na - save indices to drop same rows from y_train
    X_train_na = X_train.isna().all(axis=1)
    X_train_na_idx = X_train_na[X_train_na].index

    
    X_train_dropped = X_train.dropna()
    X_test.dropna(inplace=True)
    
    og_idx = set(X_train.index)
    dropped_idx = set(X_train_dropped.index)
    idx_to_drop = og_idx - dropped_idx
    #y_train.drop(X_train_na_idx, inplace=True)
    y_train.drop(idx_to_drop, inplace=True)
    
    X_train = X_train_dropped
    """
        
    if unvoted_recipes is not None:
        unvoted_recipes_scaled = preprocessor.transform(unvoted_recipes)
        unvoted_recipes = pd.DataFrame(unvoted_recipes_scaled, columns=unvoted_recipes.columns)
        #unvoted_recipes.dropna()
        
        
    if prediction_data is not None:
        predict_scaled = preprocessor.transform(prediction_data)
        prediction_data = pd.DataFrame(predict_scaled, columns=prediction_data.columns)
        #prediction_data.dropna()
        

    """
    # PCA for feature reduction
    # setting n_components to 157 b/c found that 95% of variance could be 
    # captured in 157 components through explained variance ratio
    # have to make lower for k fold and hp sweep to work (must be below 61)
    # interestingly it seems that auc goes down but accuracy and f1 go up with
    # 60 components
    pca = PCA(n_components=60)
    #pca = KernelPCA(n_components = 60, kernel="rbf", gamma=0.001)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    if prediction_data is not None:
        predict_pca = pca.transform(predict_scaled)

    # Calculate the cumulative sum of explained variance ratios
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Determine the number of components to reach 95% of variance
    n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
    
    # Calculate the number of features effectively 'dropped' by keeping 95% variance
    n_dropped = X.shape[1] - n_components_95
    
    print(f"Number of components to capture 95% variance: {n_components_95}")
    
    """
    # random forest for feature reduction
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    # fit to train
    forest.fit(X_train, y_train)#_scaled, y_train)
    
    # get feature importance
    feature_importances = forest.feature_importances_
    """
    Check range and values to decide threshold
    tried .001, .0001, .00001, .000001. .000001 resulted in most true positives and true negatives,
    highest accuracy and f1 for kfold CV. That is equivalent to a threshold of 0 given
    the distribution of feature importances, so optimal threshold is 0.
    
    print(f"Feature importance range: {np.min(feature_importances)} : {np.max(feature_importances)}")
    print(f"Unique feature importances: {np.unique(feature_importances)}")
    """
    
    # drop features with less than .001 importance from X_train and test
    top_feat =  np.where(feature_importances > importance_threshold)[0]
    
    X_train = X_train.iloc[:, top_feat]
    X_test = X_test.iloc[:, top_feat]
    

    if unvoted_recipes is not None:
        unvoted_recipes = unvoted_recipes.iloc[:,top_feat]#_scaled.iloc[:,top_feat]
        return X_train, X_test, unvoted_recipes, feature_importances, y_train
    else:
        return X_train, X_test, feature_importances, y_train
    """
    logic was necessary because PCA got rid of df structure, no longer the case
    with random forest
    if is_kfold == False:
        # convert the scaled arrays back to DataFrames 
        # only do this if not doing kfold, this is needed for the plot of feature 
        # importance, which isn't created when kfold is implemented 
        X_train = pd.DataFrame(X_train)#_pca) #, columns=X_train_cols.columns, index=X_train_cols.index)
        X_test = pd.DataFrame(X_test)#_pca) #, columns=X_test_cols.columns, index=X_test_cols.index)
        return X_train, X_test
    else:
        return X_train, X_test
    """
    """
        if prediction_data is not None:
            redict_pca = pd.DataFrame(predict_pca)
            return X_train, X_test, predict_pca
        else:
            return X_train, X_test
    else:
        if prediction_data is not None:
            return X_train_pca, X_test_pca, predict_pca
        else:
            return X_train_pca, X_test_pca
        """
if __name__ == "__main__":
    # define importance threshold- feature importances above this will be kept
    importance_threshold = 0
    # get user name to access votes
    user_name = input("Please enter your username\n ")
    user_path = 'cleanedVotes/'+ user_name + '.csv'
    user_votes_df = pd.read_csv(user_path, header=None, names=['title', 'vote'])
    cleaned_recipes_df = pd.read_csv('RecipeData/cleaned_recipes.csv')
    
    combined_df = pd.merge(cleaned_recipes_df, user_votes_df, on='title', how='inner')
    
    """
    trying pca instead 
    # find the important features
    selector = VarianceThreshold(threshold=0)
    selector.fit(combined_df.drop(['title', 'vote'], axis=1))
    
    # get the features that have some variance
    features_with_variance = combined_df.drop(['title', 'vote'], axis=1).columns[selector.get_support()]
    
    """
    # drop title and vote for features
    features = combined_df.drop(["title", "vote"], axis=1).columns
    # Preparing the data
    X = combined_df[features]
    y = combined_df['vote']
    

    
    # Identifying recipes not seen before (not voted on) for prediction data
    all_recipes = set(cleaned_recipes_df['title'])
    voted_recipes = set(user_votes_df['title'])
    unvoted_recipes = all_recipes - voted_recipes
    
    unvoted_recipes_df_corrected = cleaned_recipes_df[cleaned_recipes_df['title'].isin(unvoted_recipes)]
    
    # store titles for relating back later
    titles = unvoted_recipes_df_corrected["title"]
    
    # drop title
    unvoted_recipes_df_corrected = unvoted_recipes_df_corrected.drop(["title"], axis=1)
    
    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # standardize and reduce (x_train, x_test, unvoted recipes)
    X_train, X_test, unvoted_recipes, feature_importances_1, y_train = standardize_and_reduce(X_train, X_test, y_train,
                                                              importance_threshold, unvoted_recipes_df_corrected)
    
    svm_model, y_pred, y_scores = train_predict_model(
        X_train, y_train, X_test, **hyperparams)
    
    # Predicting User's preferences on these recipes
    predicted_likes_corrected = svm_model.predict(unvoted_recipes)

    
    # relate predictions back to titles
    recommended_recipes = titles[predicted_likes_corrected == 1]
    
    """
    # extracting titles of recipes predicted as likes
    recommended_recipes_titles = unvoted_recipes_df_corrected.loc[predicted_likes_corrected == 1, 'title']
    """
    
    # displaying a few recommended recipes (random selection from recommended)
    print("Recommended Recipes")
    random_rows = recommended_recipes.sample(n=5)
    print(random_rows)
    
    # plot top 10 most important features
    
    # get indices where feature importance > .001 to match X_train's reduction
    # and adjust feature importances
    important_indices = np.where(feature_importances_1 > importance_threshold)[0]
    feature_importances = feature_importances_1[important_indices]
    
    # get indices to sort feature_importances in descending order
    sorted_indices = np.argsort(feature_importances)[::-1]
    # sort
    sorted_importances = feature_importances[sorted_indices]
    
    # get names and importance
    feature_names = X_train.columns[sorted_indices]
    
    # get top 10
    top_10_features = feature_names[0:10]
    top_10_importances = sorted_importances[0:10]
  
    # magnitude, not directionality is represented here
    plt.figure()
    plt.bar(top_10_features, top_10_importances)
    plt.title('Top 10 Features Influencing User\'s Preferences')
    plt.ylabel('Importance Magnitude')
    plt.xlabel('Feature')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    
    # print accurracy and f1
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy ", accuracy)
    
    f1 = f1_score(y_test, y_pred)
    print("F1: ", f1)
    
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    print("roc auc: ", roc_auc)
    