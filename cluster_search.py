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


cleaned_recipes_df = pd.read_csv('RecipeData/cleaned_recipes.csv')

# Perhaps can take sodium and other variables into consideration in the future

# Drop all columns except the macros, titles, and calories
cleaned_recipes_df = cleaned_recipes_df[['title', 'calories', 'protein', 'fat']]

# Drop any rows with 0 calories
cleaned_recipes_df = cleaned_recipes_df[cleaned_recipes_df['calories'] > 0]


# Create a carb macro column
cleaned_recipes_df['carb'] = (cleaned_recipes_df['calories'] - 4*cleaned_recipes_df['protein'] - 9*cleaned_recipes_df['fat']) / 4



#%% Find the ratios of calories from carbs/fats/proteins
macro_ratios_df = pd.DataFrame()

macro_ratios_df['carb_ratio'] = 4 * cleaned_recipes_df['carb'] / cleaned_recipes_df['calories']


macro_ratios_df['fat_ratio'] = 9 * cleaned_recipes_df['fat'] / cleaned_recipes_df['calories']

macro_ratios_df['protein_ratio'] = 4 * cleaned_recipes_df['protein'] / cleaned_recipes_df['calories']


#%% Create a KMeans model based on the data
n_clusters = 8 # Change this to modify algorithm stinginess

kmean = KMeans(n_clusters)

kmean.fit(macro_ratios_df)

#%%
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

print(cleaned_recipes_df[cleaned_recipes_df['label'] == user_input_label].head())
