from flask import Flask, render_template, request, redirect, url_for
import os
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np




app = Flask(__name__)

DATA_PATH = 'RecipeData/cleaned_recipes.csv'
df = pd.read_csv(DATA_PATH)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/set_user', methods=['POST'])
def set_user():
    user_name = request.form['name']
    csv_path = os.path.join('cleanedVotes', f'{user_name}.csv')
    if not os.path.exists(csv_path):
        # Create a new CSV if it does not exist
        pd.DataFrame().to_csv(csv_path)
    # Redirect to another page where you can ask for macronutrient goals
    return redirect(url_for('ask_macronutrients', user_name=user_name))

@app.route('/ask_macronutrients/<user_name>')
def ask_macronutrients(user_name):
    # Render the macronutrient goals form and pass the user_name to the template
    return render_template('macros.html', user_name=user_name)

@app.route('/set_goals', methods=['POST'])
def set_goals():
    # Extract the macronutrient goals and user name from the form
    carbs = request.form['carbs']
    proteins = request.form['proteins']
    fats = request.form['fats']
    sodium = request.form['sodium']
    user_name = request.form['user_name']
    csv_path = os.path.join('cleanedGoals', f'{user_name}.csv')
    
    # Create a DataFrame to store the goals
    df = pd.DataFrame({'Carbs': [carbs], 'Proteins': [proteins], 'Fats': [fats], 'Sodium': [sodium]})
    df.to_csv(csv_path, index=False)
    
    # Redirect to a voting page, which you'll implement later
    return redirect(url_for('vote_recipes', user_name=user_name))

@app.route('/vote_recipes/<user_name>')
def vote_recipes(user_name):
    user_votes_path = os.path.join('cleanedVotes', f'{user_name}.csv')
    if os.path.exists(user_votes_path):
        user_votes_df = pd.read_csv(user_votes_path, names=['Title', 'Vote'])
        shown_recipes = set(user_votes_df['Title'])
    else:
        shown_recipes = set()

    title, ingredients = show_recipe(df, shown_recipes)
    return render_template('vote.html', title=title, ingredients=ingredients, user_name=user_name)

@app.route('/submit_vote/<user_name>/<title>', methods=['POST'])
def submit_vote(user_name, title):
    vote = request.form['vote']
    vote_val = 1 if vote == 'like' else 0
    user_votes_path = os.path.join('cleanedVotes', f'{user_name}.csv')
    with open(user_votes_path, 'a') as file:
        file.write(f"{title},{vote_val}\n")
    return redirect(url_for('vote_recipes', user_name=user_name))

# Function to show a random recipe not yet voted on
def show_recipe(df, shown_recipes):
    random_recipe = df[~df['title'].isin(shown_recipes)].sample(1).iloc[0]
    title = random_recipe['title']
    ingredients = [col for col in df.columns if random_recipe[col] == 1 and col != 'title' and col != 'rating']
    return title, ingredients


@app.route('/prepare_training/<user_name>')
def prepare_training(user_name):
    return render_template('readytotrain.html', user_name=user_name)

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


def create_model(features_df, tags_df):
    # classifier = RandomForestClassifier(random_state=0)
    classifier = DecisionTreeClassifier(random_state=0)
    
    
    x_train, x_test, y_train, y_test = train_test_split(features_df, tags_df, test_size=0.25, random_state=0)
    
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    
    return classifier, score #decision_tree_score


@app.route('/start_training/<user_name>', methods=['POST'])
def start_training(user_name):
    user_votes_path = os.path.join('cleanedVotes', f'{user_name}.csv')
    user_votes_df = pd.read_csv(user_votes_path, header=None, names=['title', 'vote'])
    
    cleaned_recipes_df = pd.read_csv(DATA_PATH)
    combined_df = pd.merge(cleaned_recipes_df, user_votes_df, on='title', how='inner')

    selector = VarianceThreshold(threshold=0)
    X = combined_df.drop(['title', 'vote'], axis=1)
    y = combined_df['vote']
    selector.fit(X)
    
    # Reduce X to the selected features.
    X_reduced = selector.transform(X)
    
    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    
    # Define hyperparameters.
    hyperparams = {'penalty': 'l2', 'C': 1.129, 'max_iter': 10000}
    
    # Train the model.
    model = LogisticRegression(**hyperparams)
    model.fit(X_train, y_train)
    
    # Find recipes not voted on by the user.
    all_recipes = set(cleaned_recipes_df['title'])
    voted_recipes = set(user_votes_df['title'])
    unvoted_recipes = all_recipes - voted_recipes
    unvoted_recipes_df = cleaned_recipes_df[cleaned_recipes_df['title'].isin(unvoted_recipes)]
    
    # Reduce the DataFrame to selected features.
    unvoted_X = unvoted_recipes_df.drop(['title'], axis=1, errors='ignore')
    print(unvoted_X)
    unvoted_X_reduced = selector.transform(unvoted_X)
    
    # Make predictions on the unvoted recipes.
    predicted_likes = model.predict_proba(unvoted_X_reduced)[:, 1]
    
    # Add predictions back to DataFrame
    unvoted_recipes_df['predicted_likes'] = predicted_likes
    
    # Sort by the probability of being liked and take the top 5.
    top_recommendations = unvoted_recipes_df.sort_values(by='predicted_likes', ascending=False).head(5)

    #DO THE TREE SEARCH
    user_goals_path = os.path.join('cleanedGoals', f'{user_name}.csv')
    user_goals_df = pd.read_csv(user_goals_path)
    user_input_df = pd.DataFrame({
        'calories': [4 * (user_goals_df['Carbs'] + user_goals_df['Proteins']) + 9 * user_goals_df['Fats']],
        'protein': user_goals_df['Proteins'],
        'fat': user_goals_df['Fats'],
        'sodium': user_goals_df['Sodium'],
        'carb': user_goals_df['Carbs']
    })


    user_output_df = user_input_df.copy()
    user_labels = []
    user_tags = []
    tag_list = []
    # tag_list.append('gourmet')
    
    # tag_list.append('side')
    # tag_list.append('dinner')
    # tag_list.append('lunch')
    # tag_list.append('dessert')
    # tag_list.append('appetizer')
    
    tag_list.append('chicken')
    tag_list.append('fish')
    tag_list.append('cod')
    tag_list.append('salmon')
    tag_list.append('beef')
    tag_list.append('beef rib')
    tag_list.append('bean')
    
    tag_list.append('pasta')
    # tag_list.append('potato')   
    
    tag_list.append('lettuce')
    tag_list.append('spinach')
    tag_list.append('bell pepper')
    
    
    forest = []
    cleaned_recipes_df, tags_df = get_cleaned_recipes(tag_list)

    total_score = 1
    for tag in tag_list:
        tag_series = tags_df[tag]
        classifier, score = create_model(cleaned_recipes_df.iloc[:,1:], tag_series)
        print(tag, score)
        
        total_score *= score
        # Put the tree in the forest
        forest.append(classifier)

    print(f'total {total_score}')

    
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
    tree_output = output_recipes_df.iloc[:,:6].head()



    
    # Pass the recommendations to a new template to display them.
    return render_template('recommendations.html', recommendations=top_recommendations, tree_output=tree_output)

if __name__ == '__main__':
    app.run(debug=True)