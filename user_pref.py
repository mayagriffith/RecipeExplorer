# playing around with logistic regression model
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

user_name = input("Please enter your username\n ")
user_path = 'cleanedVotes/'+ user_name + '.csv'
user_votes_df = pd.read_csv(user_path, header=None, names=['title', 'vote'])
cleaned_recipes_df = pd.read_csv('RecipeData/cleaned_recipes.csv')


combined_df = pd.merge(cleaned_recipes_df, user_votes_df, on='title', how='inner')

# find the most important features
selector = VarianceThreshold(threshold=0)
selector.fit(combined_df.drop(['title', 'vote'], axis=1))

# get the features that have some variance
features_with_variance = combined_df.drop(['title', 'vote'], axis=1).columns[selector.get_support()]

# Preparing the data
X = combined_df[features_with_variance]
y = combined_df['vote']

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training a logistic regression model
log_reg_model = LogisticRegression(max_iter=10000)
log_reg_model.fit(X_train, y_train)

# predicting on the test set
y_pred = log_reg_model.predict(X_test)

# evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Identifying recipes not seen before (not voted on)
all_recipes = set(cleaned_recipes_df['title'])
voted_recipes = set(user_votes_df['title'])
unvoted_recipes = all_recipes - voted_recipes

unvoted_recipes_df_corrected = cleaned_recipes_df[cleaned_recipes_df['title'].isin(unvoted_recipes)]

# Predicting Useer's preferences on these recipes
predicted_likes_corrected = log_reg_model.predict(unvoted_recipes_df_corrected[features_with_variance])

# extracting titles of recipes predicted as likes
recommended_recipes_titles = unvoted_recipes_df_corrected.loc[predicted_likes_corrected == 1, 'title']

# displaying a few recommended recipes
print(recommended_recipes_titles.head())

# find the most important features for the model
coefficients = log_reg_model.coef_[0]

# matching the coefficients with their corresponding features
feature_importance = pd.Series(coefficients, index=features_with_variance).sort_values(key=abs, ascending=False)

# selecting the top 10 most important features
top_features = feature_importance.head(10)

# plotting the feature importance
plt.figure(figsize=(10, 6))
top_features.plot(kind='bar')
plt.title('Top 10 Features Influencing User\'s Preferences')
plt.ylabel('Coefficient Value')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha="right")
plt.show()