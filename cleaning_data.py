import zipfile
import pandas as pd

zip_file = 'RecipeData.zip'  # Replace with the correct path
csv_file = 'epi_r.csv'  # Replace with the correct file name if different

# Unzipping the data
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('/Users/mayagriffith/Desktop/RecipeExplorer/RecipeData/')  # Replace with your desired path

# Load the data into a DataFrame
data_path = 'RecipeData/' + csv_file
df = pd.read_csv(data_path)

df_alcoholic = df[df['alcoholic'] == True]
print(df_alcoholic)
alcoholic_keywords = ['margarita', 'bloody mary', 'martini', 'cocktail', 'beer', 'wine', 'vodka', 'whiskey']
df_non_alcoholic = df[~df['title'].str.lower().str.contains('|'.join(alcoholic_keywords))]

columns = df.columns.tolist()
print(columns)


print('size of regular df', df.shape[0])
print('size of cleaned df', df_non_alcoholic.shape[0])

# Data Exploration
# print(df.head())
# print(df.info())
# print(df.describe())

