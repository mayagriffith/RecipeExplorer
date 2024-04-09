#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:05:27 2024

@author: Alaina Birney, Nate Gaylinn

Evaluating Model Performance
"""

import user_pref as pref_log_reg
import user_pref_svm as pref_svm
import cluster_search as cluster
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, KFold
from sklearn.metrics import (classification_report, roc_curve, auc,
                             confusion_matrix, accuracy_score, f1_score,
                             silhouette_score)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%% User Preference Model User Independent Evaluation
"""
User-independent evaluation of user preference model. Includes classification
report, ROC curve with AUC, confusion matrix. Shows that our model is slightly
better than random, but may not be the best approach due to limited data
(we have only 5 subjects). See below for LOGOCV (leave one group out CV) and Kfold
"""

def user_independent_user_pref():
    """
    Really just to make combined df because were doing k fold instead
    """
    # get data- compile all user data for user- independent evaluation
    path = 'cleanedVotes/'
    names = ["Alaina", "maya", "nate", "nick"]
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
    """
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
    # save plot
    filename = "ROC_User_Pref_Model_User-Independent_Evaluation.png"
    plt.savefig(filename)

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
    # save
    filename = "Conf_Matrix_User_Pref_Model_User-Independent_Evaluation.png"
    plt.savefig(filename)
    """

    return combined_df

#%% User Preference Model LOGOCV Evaluation
"""
LOGOCV (leave one group out cross validation) evaluation of user preference
model.

Use k fold instead.
"""
def logocv_user_pref(combined_df):
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
        log_reg_model, y_pred, y_scores = pref_log_reg.train_predict_model(X_train, y_train, X_test)

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
    plt.imshow(conf_matrix, cmap="Blues")
    plt.colorbar()
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    plt.xticks(ticks=range(len(conf_matrix)), labels=['Class 0 (Dislike)', 'Class 1 (Like)'])
    plt.yticks(ticks=range(len(conf_matrix)), labels=['Class 0 (Dislike)', 'Class 1 (Like)'])
    plt.title("Confusion Matrix Heatmap for User Preference Model with LOGOCV")
    plt.tight_layout()
    # save
    filename = "Conf_Matrix_User_Pref_Model_LOGOCV.png"
    plt.savefig(filename)

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
    # save
    filename = "ROC_User_Pref_Model_LOGOCV.png"
    plt.savefig(filename)

"""
As expected, LOGOCV shows slightly worse model preformance as it is a more
robust technique for evaluating a model when data is limited like we have
in this case.
"""

#%% User Preference Model K fold CV evaluation
"""
Independently performing CV on each user for which we have data to make assessment
more akin to real-world scenario
"""
def k_fold_CV(combined_df, reg_or_svm="svm", log_reg_args=None, svm_args = None):
    """
    Generate user-specific accuracies and f1 scores that can later be aggregated
    for an assessment of overall model performance

    Generate confusion matrix and ROC curve for aggregate results
    
    reg_or_svm should be a string, "svm" if evaluating svm user pref model,
    "reg" if evaluating log reg user pref model. The default is svm.
    """
    # drop title from whole df
    CV_df = combined_df.drop(["title"], axis=1)

    # Pass an empty dict for kwargs if none was supplied by the caller.
    if log_reg_args is None:
        log_reg_args = {}
        
    if svm_args is None:
        svm_args = {}

    # initialize lists for all metrics
    accuracies = []
    f1s = []
    true_labels = []
    pred_labels = []
    pred_scores = []

    # set num folds
    folds = 3 # using 3 bc we have a small to medium dataset

    # group by user
    grouped_df = CV_df.groupby("user_id")

    # prepare data for each group
    for user_id, group in grouped_df:
        # get features
        X = group.drop(columns=["user_id", "vote"]).values
        # get targets
        y = group["vote"].values

        # initialize K fold CV
        kf = KFold(n_splits = folds)

        # initialize lists for current user metrics
        accuracy = []
        f1 = []

        # train and predict, get metrics
        for train_idx, test_idx in kf.split(X):
            # set data for this split
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if reg_or_svm == "reg":
                log_reg_model, y_pred, y_scores = pref_log_reg.train_predict_model(
                    X_train, y_train, X_test, **log_reg_args)
            if reg_or_svm == "svm":
                X_train, X_test = pref_svm.standardize(X_train, X_test, is_kfold=True)
                svm_model, y_pred, y_scores = pref_svm.train_predict_model(
                    X_train, y_train, X_test, **svm_args)
            
            # train and predict - 

            # evaluate- get metrics
            ind_accuracy = accuracy_score(y_test, y_pred)
            ind_f1 = f1_score(y_test, y_pred)

            # store metrics
            accuracy.append(ind_accuracy)
            f1.append(ind_f1)

            # store labels and scores
            true_labels.extend(y_test)
            pred_labels.extend(y_pred)
            pred_scores.extend(y_scores)

        # store avg user results in overall list
        accuracies.append(np.mean(accuracy))
        f1s.append(np.mean(f1))

    return accuracies, f1s, true_labels, pred_labels, pred_scores


def visualize_k_fold_CV(accuracies, f1s, true_labels, pred_labels, pred_scores):
    # compute and plot confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print("User Pref KFold Aggregate Confusion Matrix\n", conf_matrix)
    plt.figure()
    plt.imshow(conf_matrix, cmap="Blues")
    plt.colorbar()
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    plt.xticks(ticks=range(len(conf_matrix)), labels=['Class 0 (Dislike)', 'Class 1 (Like)'])
    plt.yticks(ticks=range(len(conf_matrix)), labels=['Class 0 (Dislike)', 'Class 1 (Like)'])
    plt.title("Aggregate Confusion Matrix Heatmap for User Preference Model with KFold CV")
    plt.tight_layout()
    # save
    filename = "Conf_Matrix_User_Pref_Model_KFold.png"
    plt.savefig(filename)
    plt.show()
    plt.close()

    # compute and plot ROC curve with AUC
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 4)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Aggregate ROC Curve for User Preference Model with KFold CV")
    plt.legend(loc="lower right")
    # save
    filename = "ROC_User_Pref_Model_KFold.png"
    plt.savefig(filename)
    plt.show()
    plt.close()


#%% Train vs test error for user preference model
"""
Easy way to tell if our model is over or underfitting to have more indication
of next steps
"""
def train_test_user_pref(combined_df, reg_or_svm="svm", log_reg_args = None, svm_args = None):
    """
    reg_or_svm should be a string, "svm" if evaluating svm user pref model,
    "reg" if evaluating log reg user pref model. The default is svm.
    """
    CV_df = combined_df.drop(["title"], axis=1)
    # Features
    X = CV_df.drop(columns=["user_id","vote"]).values
    # target
    y = CV_df["vote"].values
    
    # Pass an empty dict for kwargs if none was supplied by the caller.
    if log_reg_args is None:
        log_reg_args = {}
        
    if svm_args is None:
        svm_args = {}

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

        # feature selection
        selector = VarianceThreshold(threshold=0)
        X_train = selector.fit_transform(X_train)
        X_test = selector.transform(X_test)
        
        if reg_or_svm == "reg":
            model, y_pred, y_scores = pref_log_reg.train_predict_model(X_train, y_train, X_test, **log_reg_args)
        if reg_or_svm == "svm":
            model, y_pred, y_scores = pref_svm.train_predict_model(X_train, y_train, X_test, **svm_args)

        # get and store errors
        # using 1-accuracy because classification with balanced dataset
        train_error = 1 - accuracy_score(y_train, model.predict(X_train))
        train_error_all.append(train_error)

        test_error = 1 - accuracy_score(y_test, y_pred)
        test_error_all.append(test_error)
    
    if reg_or_svm == "svm":
        model_type = "SVM"
    if reg_or_svm == "reg":
        model_type = "Logistic Regression"
    # plot
    plt.figure()
    plt.plot(train_sizes*100, train_error_all, label="Train")
    plt.plot(train_sizes*100, test_error_all, label="Test")
    plt.xlabel("Training Data Used")
    plt.ylabel("Error (1-Accuracy)")
    plt.title(f"Train vs. Test Error For User Preference Model ({model_type})")
    plt.legend()
    plt.show()
    # save
    filename = f"Train_Test_User_Pref_Model_{model_type}.png"
    plt.savefig(filename)

#%% K Means Recommendation Model Silhouette Score Evaluation

def silhouette_score_recommendation(clusters_range, macro_ratios_df):
    # calculate silhouette score for various numbers of clusters
    # initialize list to store scores
    sil_scores = []
    # create model for clusters in range
    for cluster_num in clusters_range:
        kmean, labels = cluster.create_model(cluster_num, macro_ratios_df,
                                             purpose="Sil_score")
        score = silhouette_score(macro_ratios_df, labels)
        sil_scores.append(score)

    # Plot
    plt.figure()
    plt.plot(clusters_range, sil_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Various Numbers of Clusters')
    plt.show()
    # save
    filename = "Sil_Score_Recommendation_Model.png"
    plt.savefig(filename)
    return sil_scores, macro_ratios_df

#%% K Means Recommendation Model Visualization

# reduce dimensionality for plotting with t-SNE (b/c relationships arent linear)
# 3D, want to get down to 2D
def visualize_recommendation_tsne(macro_ratios_df):
    # standardize
    scalar = StandardScaler()
    macro_ratios_df_stand = scalar.fit_transform(macro_ratios_df)
    # TSNE initialization
    tsne = TSNE(random_state=42)
    # TSNE application
    results = tsne.fit_transform(macro_ratios_df_stand)
    # create k means model for reduced data
    # using 4 clusters as indicated by sil score and standardized data
    # for better compatibility with tsne
    kmeans = cluster.create_model(4,macro_ratios_df_stand)
    cluster_labels = kmeans.labels_
    # plot clusters with reduced data
    num_clusters_tsne = len(set(cluster_labels))
    plt.figure()
    for i in range(num_clusters_tsne):
        tsne_cluster = results[cluster_labels == i]
        plt.scatter(tsne_cluster[:,0], tsne_cluster[:,1], label=f"Cluster {i+1}", s=10)
    plt.title("t-SNE Reduced Dimensionality Visualization of Clusters")
    plt.legend(title="Cluster Labels")
    plt.tight_layout()
    plt.show()
    # save
    filename = "Visualization_Recommendation_Clusters.png"
    plt.savefig(filename)

def visualize_recommendation_3D(macro_ratios_df):
    # create k means model
    # using 4 clusters as indicated by sil score
    kmeans = cluster.create_model(4,macro_ratios_df)
    cluster_labels = kmeans.labels_
    num_clusters_3D = len(set(cluster_labels))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(num_clusters_3D):
        cluster_idx = np.where(cluster_labels == i)
        ax.scatter(macro_ratios_df.iloc[cluster_idx]["carb_ratio"],
                    macro_ratios_df.iloc[cluster_idx]["fat_ratio"],
                    macro_ratios_df.iloc[cluster_idx]["protein_ratio"], label=f"Cluster {i+1}")
    plt.title("3D Visualization of Clusters for K-Means Recommendation Model")
    ax.set_xlabel("Carb Ratio")
    ax.set_ylabel("Fat Ratio")
    ax.set_zlabel("Protein Ratio")
    ax.legend(title="Cluster Labels")
    plt.tight_layout()
    plt.show()

#%% Run

if __name__ == '__main__':
    # User Preference Model
    # Get ROC with AUC, conf matrix, and classification report for user preference
    # model with user independent evaluation
    combined_df = user_independent_user_pref()
    # this was all logocv method
    # Get ROC with AUC, conf matrix, average accuracy, and average F1 score for
    #user preference model with logocv
    #logocv_user_pref(combined_df)


    # kfold method
    model_type = input("Are you evaluating the logistic regression or svm user preference model? (enter reg or svm)")
    
    if model_type == "reg":
        kfold_results = k_fold_CV(combined_df, model_type, pref_log_reg.hyperparams, pref_svm.hyperparams) # these are per user
        #Get plot of train v test error for user preference model
        train_test_user_pref(combined_df, model_type, pref_log_reg.hyperparams, pref_svm.hyperparams)
    if model_type == "svm":
        kfold_results = k_fold_CV(combined_df, model_type, pref_log_reg.hyperparams, pref_svm.hyperparams) # these are per user
        #Get plot of train v test error for user preference model
        train_test_user_pref(combined_df, model_type, pref_log_reg.hyperparams, pref_svm.hyperparams)
    accuracies, f1s, _, _, _ = kfold_results
    visualize_k_fold_CV(*kfold_results)

    # average metrics across users
    avg_accuracy = np.mean(accuracies)
    avg_f1 = np.mean(f1s)
    print("accuracy:", avg_accuracy)
    print("f1:", avg_f1)

    """
    User preference next steps
    Overall, this model is a good starting point but there is certainly room for
    improvement as it is only slightly better than random. Next steps: investigate
    ways to improve feature selection, implement regularization using random
    search to fine tune hyperparam C and trying penalties other than L2/ ridge,
    attempt to get more data.

    Results of KFold CV were similar to LOGOCV
    """

    # K means recommendation model
    # get data
    """
    cleaned_recipes_df = cluster.get_cleaned_recipes()
    macro_ratios_df = cluster.get_macro_ratios(cleaned_recipes_df)
    # Silhouette Score plot for recommendation model
    # set cluster range to try
    clusters_range = range(3,17)
    sil_scores, macro_ratios_df = silhouette_score_recommendation(clusters_range, macro_ratios_df)
    """
    """
    showing that optimal # clusters is 4, but still only has a sil score of .42,
    indicating that there is room for improvement.
    """
    # Visualization of clusters
    # this one (below) is hard to interpret
    #visualize_recommendation_3D(macro_ratios_df)
    # warning this (below) takes a while
    #visualize_recommendation_tsne(macro_ratios_df)
    """
    Clusters are close together, a likely explanation for why our max silhouette
    score (for 4 clusters, which is the number being used for this visualization)
    was .47 instead of closer to 1
    """
    #print(combined_df.head())
