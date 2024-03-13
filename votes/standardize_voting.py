import csv
import pandas as pd
import os
import re

def remove_quotes(text):
    return re.sub(r'[“”\"\'‘’]', '', str(text))

def clean_vote_value(value):
    try:
        return int(float(value))
    except ValueError:
        return None

def parse_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, quotechar='"')
        data = []
        for row in reader:
            if len(row) != 2:
                print(f"Problematic line: {row}")
                # correct lines with more than 2 fields
                corrected_title = ','.join(row[:-1]).strip('"')
                corrected_vote = clean_vote_value(row[-1])
                if corrected_vote is not None:
                    print(corrected_title)
                    data.append([corrected_title, corrected_vote])
            else:
                title = remove_quotes(row[0].strip())
                vote = clean_vote_value(row[1])
                if vote is not None:
                    data.append([title, vote])
    return pd.DataFrame(data, columns=['title', 'vote'])

votes_dir = 'votes'
cleaned_votes_dir = 'cleanedVotes'

if not os.path.exists(cleaned_votes_dir):
    os.makedirs(cleaned_votes_dir)

# iterate over each uncleaned vote
for file_name in os.listdir(votes_dir):
    if file_name.endswith('.csv'):  # Process only CSV files
        file_path = os.path.join(votes_dir, file_name)
        cleaned_file_path = os.path.join(cleaned_votes_dir, file_name)
        
        # print problematic lines, and save the cleaned data
        df = parse_csv(file_path)
        df.to_csv(cleaned_file_path, index=False, header=False)

print("Voting files cleaned, standardized, and saved to 'cleanedVotes' folder.")