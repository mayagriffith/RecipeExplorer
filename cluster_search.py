'''
cluster search will (ideally) sort the existing recipes into K groups based on
the proportions of cal_protein / calories, cal_fat / calories, and cal_carb / calories

Using the estimation of the FNIC (subdivision(?) of the USDA):
    Carbohydrates provide 4 calories per gram,
    protein provides 4 calories per gram,
    and fat provides 9 calories per gram.
'''

#%% Setup

import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np

def get_cleaned_recipes():
    cleaned_recipes_df = pd.read_csv('RecipeData/cleaned_recipes.csv')
    
    # Perhaps can take sodium and other variables into consideration in the future
    
    # Drop all columns except the macros, titles, and calories
    cleaned_recipes_df = cleaned_recipes_df[['title', 'calories', 'protein', 'fat']]
    
    # Drop any rows with 0 calories
    cleaned_recipes_df = cleaned_recipes_df[cleaned_recipes_df['calories'] > 0]
    
    
    # Create a carb macro column
    cleaned_recipes_df['carb'] = (cleaned_recipes_df['calories'] - 4*cleaned_recipes_df['protein'] - 9*cleaned_recipes_df['fat']) / 4
    return cleaned_recipes_df




#%% Find the ratios of calories from carbs/fats/proteins
def get_macro_ratios(cleaned_recipes_df):
    macro_ratios_df = pd.DataFrame()
    
    macro_ratios_df['carb_ratio'] = 4 * cleaned_recipes_df['carb'] / cleaned_recipes_df['calories']
    
    
    macro_ratios_df['fat_ratio'] = 9 * cleaned_recipes_df['fat'] / cleaned_recipes_df['calories']
    
    macro_ratios_df['protein_ratio'] = 4 * cleaned_recipes_df['protein'] / cleaned_recipes_df['calories']
    return macro_ratios_df



#%% Create a KMeans model based on the data
n_clusters = 4 # Change this to modify algorithm stinginess- using 4 b/c best silhouette score

# default purpose is recommending, so nothing needs to be added for purpose
# when this model is being used for recommendations. Must specify purpose = 
# Sil_score to easily get silhouette score
def create_model(n_clusters, macro_ratios_df, purpose="Recommending"):
    kmean = KMeans(n_clusters)
    if purpose == "Recommending":
        kmean.fit(macro_ratios_df)
    elif purpose == "Sil_score":
        labels = kmean.fit_predict(macro_ratios_df)
    else:
        print("Purpose must either be 'Recommending' or 'Sil_score'")
    if purpose == "Sil_score":
        return kmean, labels
    else:
        return kmean



#%%
if __name__ == "__main__":
    cleaned_recipes_df = get_cleaned_recipes()
    macro_ratios_df = get_macro_ratios(cleaned_recipes_df)
    kmean = create_model(n_clusters, macro_ratios_df)
    
    
    cleaned_recipes_df['label'] = kmean.labels_ # This will give which recipes belong to each label
    
    
    
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
