import zipfile
import pandas as pd

zip_file = 'RecipeData.zip' 
csv_file = 'epi_r.csv' 

# Unzipping the data
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('/Users/mayagriffith/Desktop/RecipeExplorer/RecipeData/')  # Replace with your desired path

# Load the data into a DataFrame
data_path = 'RecipeData/' + csv_file
df = pd.read_csv(data_path)

df_clean = df[df['alcoholic'] == False]
columns = df.columns.tolist()
print(columns)


print('size of regular df', df.shape[0])
print('size of cleaned df', df_clean.shape[0])

# Data Exploration
# print(df.head())
# print(df.info())
# print(df.describe())

