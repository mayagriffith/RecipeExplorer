'''
Tree search branching off from the tree search framework.
Uses a subset of tags as the labels.
The first model will use EVERY tag - with tags ideally
pruned if they have limited significant correlation to
the features.

Using the estimation of the FNIC (subdivision of the USDA):
    Carbohydrates provide 4 calories per gram,
    protein provides 4 calories per gram,
    and fat provides 9 calories per gram.
'''

# Setup

import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
import numpy as np

def get_cleaned_recipes(tag_list):
    cleaned_recipes_df = pd.read_csv('RecipeData/cleaned_recipes.csv')
    
    # Perhaps can take sodium and other variables into consideration in the future
    cleaned_recipes_df = cleaned_recipes_df#.iloc[:3,:]
    
    # Drop any rows with 0 calories
    cleaned_recipes_df = cleaned_recipes_df[cleaned_recipes_df['calories'] > 0]
    

    tag_cols = cleaned_recipes_df[tag_list]
    # tag_cols = cleaned_recipes_df.iloc[:, 5:]# All tags version
    
    # Try with the tag_cols outright
    # # Get the number of tags to define the string length for the labels
    # num_tags = len(tag_cols.iloc[0])
    
    # # Initialize a concatenated column
    # concat_tag_cols = np.zeros(tag_cols.iloc[:,0].shape, dtype=f'U{num_tags}') #pd.DataFrame(np.zeros(tag_cols.iloc[:,0].shape, dtype=str))
    
    # A bit unprofessional, but it gets there
    # for column in tag_cols:
    #     for row_index in range(len(concat_tag_cols)):
    #         if tag_cols[column].iloc[row_index] == 0.0:
    #             concat_tag_cols[row_index] += '0'
    #         else:
    #             concat_tag_cols[row_index] += '1'
                
    # Drop all columns except the macros, titles, and calories
    cleaned_recipes_df = cleaned_recipes_df[['title', 'calories', 'protein', 'fat', 'sodium']]
    
    # Create a carb macro column
    cleaned_recipes_df['carb'] = (cleaned_recipes_df['calories'] - 4*cleaned_recipes_df['protein'] - 9*cleaned_recipes_df['fat']) / 4
    
    return cleaned_recipes_df, tag_cols #, tag_list

# Create a decision tree model based on the data


def create_model(features_df, tags_df):
    # classifier = RandomForestClassifier(random_state=0)
    classifier = DecisionTreeClassifier(random_state=0)
    
    
    x_train, x_test, y_train, y_test = train_test_split(features_df, tags_df, test_size=0.25)
    
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    
    return classifier, score #decision_tree_score



#
if __name__ == "__main__":
    # Specific tag columns to look at:
    tag_list = []
    # tag_list.append('gourmet')
    
    tag_list.append('side')
    tag_list.append('dinner')
    tag_list.append('lunch')
    tag_list.append('dessert')
    tag_list.append('appetizer')
    
    tag_list.append('chicken')
    tag_list.append('fish')
    tag_list.append('cod')
    tag_list.append('salmon')
    tag_list.append('beef')
    tag_list.append('beef rib')
    tag_list.append('bean')
    
    
    # tag_list.append('bread')
    tag_list.append('pasta')
    tag_list.append('cake')
    

    # tag_list.append('tomato')
    # tag_list.append('onion')
    tag_list.append('potato')   
    # tag_list.append('turnip')
    
    
    # tag_list.append('low cal')
    tag_list.append('lettuce')
    tag_list.append('spinach')
    tag_list.append('bell pepper')
    
    
    forest = []
    
    
    cleaned_recipes_df, tags_df = get_cleaned_recipes(tag_list)
    
    for tag in tag_list:
        tag_series = tags_df[tag]
        # macro_ratios_df = get_macro_ratios(cleaned_recipes_df)
        # print(create_model(cleaned_recipes_df.iloc[:,1:], tags_df))
        classifier, score = create_model(cleaned_recipes_df.iloc[:,1:], tag_series)
        print(tag, score)
        
        # cleaned_recipes_df['label'] = kmean.labels_ # This will give which recipes belong to each label
        forest.append(classifier)

    #%%
    
    # Get user input
    print("First, please input your prefered macro amount") # asking for amount here, but it is really more like the ratios
    carb_goal = int(input("Please enter your desired carb amount(g): "))
    fat_goal = int(input("Please enter your desired fat amount(g): "))
    protein_goal = int(input("Please enter your desired protein amount(g): "))
    sodium_goal = int(input("Please enter your desired sodium amount (mg): "))
    
    calorie_goal = 4 * (carb_goal + protein_goal) + 9 * fat_goal
    
    # Condense the user choice into a DF
    user_input_df = pd.DataFrame()
    user_input_df['calories'] = [calorie_goal]
    user_input_df['protein'] = [protein_goal]
    user_input_df['fat'] =  [fat_goal]
    user_input_df['sodium'] = [sodium_goal]
    user_input_df['carb'] = [carb_goal]
    
    user_labels = []
    user_tags = []
    for tag_index, classifier in enumerate(forest):
        # Extract the label from the prediction
        user_input_label = classifier.predict(user_input_df)[0]
        user_input_label = user_input_label == 1
        user_labels.append(user_input_label)
        if user_input_label:
            user_tags.append(tag_list[tag_index])
        
    print('\n\n\n')
    print('Tags:', user_tags)
    
    # make it so that all columns are displayed so protein and fat can be viewed easily
    pd.set_option('display.max_columns', None)
    
    # Likely a better way than this, but this isn't that hard
    recipe_match = []
    for i in range(len(tags_df)):
        recipe_match.append((tags_df.iloc[i,:] == user_labels).all())
    
    print(cleaned_recipes_df[recipe_match].head())
    