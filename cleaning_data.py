import zipfile
import pandas as pd
import os
import re

#function to remove quotes from the titles of the dataframe
def remove_quotes(text):
    return re.sub(r"[“”\"'‘’]", '', str(text))

# Unzipping and loading the data
zip_file = 'RecipeData.zip' 
csv_file = 'epi_r.csv' 
current_directory = os.getcwd()
extract_path = os.path.join(current_directory, "RecipeData")

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

data_path = os.path.join(extract_path, csv_file)
df = pd.read_csv(data_path)

# taking drinks out of the dataset
df_clean = df[(df['alcoholic'] == False) & (df['non-alcoholic'] == False) & (df['cocktail'] == False)]
# taking duplicate recipes out of the dataset
df_clean = df_clean.dropna().drop_duplicates(subset='title')
# remove quotes from dataset
df_clean['title'] = df_clean['title'].str.strip().apply(remove_quotes)

# Saving the cleaned data

#dropping the top 7 caloric recipes b/c they look like outliers
top_7_caloric_recipes = df_clean.sort_values(by='calories', ascending=False).head(7)

df_clean = df_clean.drop(top_7_caloric_recipes.index)


# dropping double 'bon appietit' column
column_to_drop = df_clean.columns[64]
df_clean = df_clean.drop(column_to_drop, axis=1)


df_clean.reset_index(drop=True, inplace=True)


# Print the size of dataframes and some cleaned titles for verification
print('size of regular df:', df.shape[0])
print('size of cleaned df:', df_clean.shape[0])


csv_output_file = 'cleaned_recipes.csv'
df_clean.to_csv(os.path.join(extract_path, csv_output_file), index=False)

