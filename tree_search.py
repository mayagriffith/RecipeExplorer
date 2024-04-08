'''
Tree search branching off from the cluster search framework.
Uses a concatenation of a subset of tags as the label.
The first model will use EVERY tag - with tags ideally
pruned if they have limited significant correlation to
the features.

Using the estimation of the FNIC (subdivision(?) of the USDA):
    Carbohydrates provide 4 calories per gram,
    protein provides 4 calories per gram,
    and fat provides 9 calories per gram.
'''

#%% Setup

import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import numpy as np

def get_cleaned_recipes():
    cleaned_recipes_df = pd.read_csv('RecipeData/cleaned_recipes.csv')
    
    # Perhaps can take sodium and other variables into consideration in the future
    cleaned_recipes_df = cleaned_recipes_df#.iloc[:3,:]
    
    # Drop any rows with 0 calories
    cleaned_recipes_df = cleaned_recipes_df[cleaned_recipes_df['calories'] > 0]
    
    # All rows, only tag cols (six and onwards)
    tag_cols = cleaned_recipes_df.iloc[:, 5:]#.astype(str)
    
    # Get the number of tags to define the string length for the labels
    num_tags = len(tag_cols.iloc[0])
    
    # Initialize a concatenated column
    concat_tag_cols = np.zeros(tag_cols.iloc[:,0].shape, dtype=f'U{num_tags}') #pd.DataFrame(np.zeros(tag_cols.iloc[:,0].shape, dtype=str))
    
    # A bit unprofessional, but it gets there
    for column in tag_cols:
        for row_index in range(len(concat_tag_cols)):
            if tag_cols[column].iloc[row_index] == 0.0:
                concat_tag_cols[row_index] += '0'
            else:
                concat_tag_cols[row_index] += '1'
                
    # Drop all columns except the macros, titles, and calories
    cleaned_recipes_df = cleaned_recipes_df[['title', 'calories', 'protein', 'fat', 'sodium']]
    
    # Create a carb macro column
    cleaned_recipes_df['carb'] = (cleaned_recipes_df['calories'] - 4*cleaned_recipes_df['protein'] - 9*cleaned_recipes_df['fat']) / 4
    
    
    cleaned_recipes_df['tags'] = concat_tag_cols
    
    return cleaned_recipes_df




#%% Find the ratios of calories from carbs/fats/proteins
def get_macro_ratios(cleaned_recipes_df):
    macro_ratios_df = pd.DataFrame()
    
    macro_ratios_df['carb_ratio'] = 4 * cleaned_recipes_df['carb'] / cleaned_recipes_df['calories']
    
    
    macro_ratios_df['fat_ratio'] = 9 * cleaned_recipes_df['fat'] / cleaned_recipes_df['calories']
    
    macro_ratios_df['protein_ratio'] = 4 * cleaned_recipes_df['protein'] / cleaned_recipes_df['calories']
    return macro_ratios_df



#%% Create a decision tree model based on the data


def create_model(df):
    dtc = DecisionTreeClassifier(random_state=0)
    
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns!='tags'],df['tags'], test_size=0.25, random_state = 0)# add your code here.
    
    dtc.fit(x_train, y_train)
    decision_tree_score = dtc.score(x_test, y_test)
    
    
    return decision_tree_score



#%%
if __name__ == "__main__":
    cleaned_recipes_df = get_cleaned_recipes()
    
    # Implement the ratios later? Or maybe not at all
    # macro_ratios_df = get_macro_ratios(cleaned_recipes_df)
    print(create_model(cleaned_recipes_df.iloc[:,1:]))
    
    
    # cleaned_recipes_df['label'] = kmean.labels_ # This will give which recipes belong to each label
    
    
    
    #%% Get user input
    print("First, please input your prefered macro amount") # asking for amount here, but it is really more like the ratios
    carb_goal = int(input("Please enter your desired carb amount: "))
    fat_goal = int(input("Please enter your desired fat amount: "))
    protein_goal = int(input("Please enter your desired protein amount: "))
    
    calorie_goal = 4 * (carb_goal + protein_goal) + 9 * fat_goal
    
    carb_goal_ratio = 4 * carb_goal / calorie_goal
    fat_goal_ratio = 9 * fat_goal / calorie_goal
    protein_goal_ratio = 4 * protein_goal / calorie_goal
    
    #%%
    
    user_input_df = pd.DataFrame()
    user_input_df['carb_ratio'] = [carb_goal_ratio]
    user_input_df['fat_ratio'] =  [fat_goal_ratio]
    user_input_df['protein_ratio'] = [protein_goal_ratio]
    #%%
    
    # Extract the label from the prediction
    user_input_label = kmean.predict(user_input_df)[0]
    # make it so that all columns are displayed so protein and fat can be viewed easily
    pd.set_option('display.max_columns', None)
    print(cleaned_recipes_df[cleaned_recipes_df['label'] == user_input_label].head())
