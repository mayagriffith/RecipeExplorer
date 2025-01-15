# RecipeExplorer


Run cleaning_data.py and this will extract RecipeData.zip, clean the files and add an extra clean csv to reference.
It is also zipped for your convience, but some directories may need to be modified

Run votes/standardize_votes.py to get clean votes, that are able to be used as data to train on the model

User_pref.py is a logistic regression model that takes a user's votes into account and will give recipes that most fit their preference model. Must run votes/standardize_votes.py to get clean data for this to use.

Create_votes.py allows a user to vote on random recipes, that will be used as data in the user preference model.

Evaluate.py allows a user to view performance metrics for the user preference and recommendation models. For the user preference model: includes user independent evaluation as well as leave-one-group-out cross-validation, average accuracy and F1 scores over folds, ROC curve and ROC-AUC calculation, and confusion matrix visualization. For the recommendation model: includes silhouette score calculation and t-SNE reduced dimensionality visualization of clusters.
