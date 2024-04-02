import pandas as pd
import random
import os

# A per-category target for the number of votes.
TARGET = 100

current_directory = os.getcwd()
extract_path = os.path.join(current_directory, "RecipeData")
csv_file = 'cleaned_recipes.csv'
data_path = os.path.join(extract_path, csv_file)
df = pd.read_csv(data_path)

def show_recipe(df, shown_recipes):
    random_recipe = df.sample(1).iloc[0]
    while random_recipe['title'] in shown_recipes:
        random_recipe = df.sample(1).iloc[0]  # Keep sampling until you get a new recipe

    title = random_recipe['title']
    ingredients = [col for col in df.columns if random_recipe[col] == 1 and col != 'title' and col != 'rating']
    
    print("Recipe: " + title)
    print("Ingredients:")
    for ingredient in ingredients:
        print("- " + ingredient)

    shown_recipes.add(title)  # Add the title to the set of shown recipes
    return title, ingredients

def vote_recipe(df):
    liked_recipes = []
    disliked_recipes = []
    shown_recipes = set()  # Set to keep track of recipes that have been voted on

    while len(liked_recipes) < TARGET or len(disliked_recipes) < TARGET:
        title, ingredients = show_recipe(df, shown_recipes)
        vote = input("Do you like this recipe? Enter 1 for Yes, 0 for No: ")

        if vote == '1' and len(liked_recipes) < TARGET:
            liked_recipes.append((title, 1))
        elif vote == '0' and len(disliked_recipes) < TARGET:
            disliked_recipes.append((title, 0))
        else:
            print("Invalid input or maximum votes reached for the category.")

        # Show progress
        print(f"Liked: {len(liked_recipes)} Disliked: {len(disliked_recipes)}")

    return liked_recipes + disliked_recipes  # Combine the lists

def get_name():
    name = input("What is your name?")
    return name

def main():
    # get name
    name = get_name()
    # Collect votes
    recipes = vote_recipe(df)

    recipe_df = pd.DataFrame(recipes, columns=['Title', 'Vote'])

    # Save to CSV without column headers
    recipe_df.to_csv(f'cleanedVotes/{name}.csv', index=False, header=False)

    print(f"Voting complete. Data saved to {name}.csv.")

# Check if the script is the main program and run it
if __name__ == "__main__":
    main()
