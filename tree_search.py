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
                
    # Drop all columns except the macros, titles, and calories
    cleaned_recipes_df = cleaned_recipes_df[['title', 'calories', 'protein', 'fat', 'sodium']]
    
    # Create a carb macro column
    cleaned_recipes_df['carb'] = (cleaned_recipes_df['calories'] - 4*cleaned_recipes_df['protein'] - 9*cleaned_recipes_df['fat']) / 4
    
    return cleaned_recipes_df, tag_cols #, tag_list

# Create a decision tree model based on the data


def create_model(features_df, tags_df, test_size=0.2):
    # classifier = RandomForestClassifier(random_state=0)
    classifier = DecisionTreeClassifier(random_state=0, max_leaf_nodes=75)
    
    
    x_train, x_test, y_train, y_test = train_test_split(features_df, tags_df, test_size=test_size, random_state=0)
    
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    # score_train = classifier.score(x_train, y_train)
    
    return classifier, score#, score_train #decision_tree_score



#
if __name__ == "__main__":
    # Specific tag columns to look at:
    tag_list = []
    
    tag_list.append('chicken')
    tag_list.append('fish')
    tag_list.append('cod')
    tag_list.append('salmon')
    tag_list.append('beef')
    tag_list.append('beef rib')
    tag_list.append('bean')
    
    tag_list.append('pasta')
    
    tag_list.append('lettuce')
    tag_list.append('spinach')
    tag_list.append('bell pepper')
    
    
    forest = []
    
    
    cleaned_recipes_df, tags_df = get_cleaned_recipes(tag_list)
    total_score = 1
    for tag in tag_list:
        tag_series = tags_df[tag]
        classifier, score = create_model(cleaned_recipes_df.iloc[:,1:], tag_series, 0.2)
        print(tag, score)
        
        total_score *= score
        # Put the tree in the forest
        forest.append(classifier)
        
    print(f'total {total_score}')
    
    # total_errors = []
    # total_errors_train = []
    # train_sizes = np.arange(0.01,1.0,0.05)
    # # Use to retrieve train vs. test error plot
    # for train_size in train_sizes:
        
    #     forest = []
        
        
    #     cleaned_recipes_df, tags_df = get_cleaned_recipes(tag_list)
    #     total_score = 1
    #     total_score_train = 1
    #     for tag in tag_list:
    #         tag_series = tags_df[tag]
    #         classifier, score, score_train = create_model(cleaned_recipes_df.iloc[:,1:], tag_series, 1-train_size)
    #         print(tag, score)
            
    #         total_score *= score
    #         total_score_train *= score_train
    #         # Put the tree in the forest
    #         forest.append(classifier)
            
    #     total_errors.append(1-total_score)
    #     total_errors_train.append(1-total_score_train)
    #     print(f'total {total_score}')
    
    # plt.figure(2,clear=True)
    # plt.plot(train_sizes*100, total_errors_train, label='Train')
    # plt.plot(train_sizes*100, total_errors, label='Test')
    
    # plt.xlabel('Training Data Used')
    # plt.ylabel('Error (1-Accuracy)')
    # plt.legend()
    # plt.title('Test Error vs. Training Data Levels Forest')
    
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
    
    user_output_df = user_input_df.copy()
    
    user_labels = []
    user_tags = []
    for tag_index, classifier in enumerate(forest):
        # Extract the label from the prediction
        user_input_label = classifier.predict(user_input_df)[0]
        
        
        user_output_df[tag_list[tag_index]] = [user_input_label] # This will give the tag to the users item
        
        user_input_label = user_input_label == 1
        user_labels.append(user_input_label)
        
        # Get list for printing
        if user_input_label:
            user_tags.append(tag_list[tag_index])
        
    print('\n\n\n')
    print('Tags:', user_tags)
    
    # make it so that all columns are displayed so protein and fat can be viewed easily
    pd.set_option('display.max_columns', None)
    
   
    output_recipes_df = cleaned_recipes_df.join(tags_df)
    
    # Find the recipes that match all assumed tags, and only those tags
    for tag in tag_list:
        tag_value = user_output_df[tag][0]
        print(tag_value)
        output_recipes_df = output_recipes_df[output_recipes_df[tag] == tag_value]
    
    # Print the recipes and 
    print(output_recipes_df.iloc[:,:6].head())
    