# imports
import sys
import sqlite3 # note -- for running on AWS, this requires a different configuration
import pandas as pd
import pickle
import chromadb
from chromadb import HttpClient
from chromadb.config import Settings
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from chromadb.utils import embedding_functions
from sklearn.model_selection import train_test_split
import os

# globals
chroma_source = "ENTER_CHROMA_URL_HERE" # can be local or hosted
home_directory = "DIRECTORY_TO_FILES"
BIGRAG_embeddings = ""
BIGRAG_full = ""
BIGRAG_name = "ONNX"
path_personality_news = "path_to_human_data_for_ranking"
path_personality_persuasion = "path_to_human_data_for_ranking"
personality_data_news = pd.read_csv(path_personality_news)
personality_data_persuasion = pd.read_csv(path_personality_persuasion)
dataset1 = personality_data_news
features = ["features", "of", "data", "here"]
dataset2 = personality_data_persuasion
with open("path_to_pickle_file", 'rb') as f:
    BIGRAG_embeddings = pickle.load(f)
chunksize=1000
tfr = pd.read_csv("path_to_RAG_data", chunksize=chunksize, iterator=True)
BIGRAG_full = pd.concat(tfr, ignore_index=True)

'''
BEGIN NEWS RANKING
'''

def get_train_test_data_news(df, feature_columns,target_columns, test_size=0.0, random_state=42):
    # Split the data into training and test sets
    train_df = df
    test_df = None
    selected_columns = ['user_id'] + feature_columns + target_columns
    train_df = train_df[selected_columns]

    # Extract features based on the passed feature_columns
    train_df_features = train_df[feature_columns]
    test_df_features = None

    # Create user-item matrices
    train_user_item_matrix = pd.concat([train_df.iloc[:, 0], train_df.iloc[:, -10:]], axis=1)
    test_user_item_matrix = None
    return train_df, test_df, train_df_features, test_df_features, train_user_item_matrix, test_user_item_matrix

def get_top_news_sources_per_person(df, num_top_sources):
    """
    Function to find the top news sources for each person based on their scores.

    Parameters:
    - df: pandas DataFrame where each row represents a user and each column (except 'user_id') represents a news source score.
    - num_top_sources: Number of top news sources to select per person (default is 3).

    Returns:
    - top_news_sources_df: A DataFrame with each user's ID and their top news sources.
    """
    # Initialize a list to store the top news sources per person
    top_news_sources_list = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Exclude the 'user_id' column and get the rest as a Series
        news_sources = row.drop('user_id')

        # Sort the news sources by score in descending order and get the top ones
        top_news_sources = news_sources.sort_values(ascending=False).index.tolist()[:num_top_sources]

        # Create a dictionary to store the user_id and their top news sources
        result = {
            'user_id': row['user_id'],
            'top_news_sources': top_news_sources
        }

        # Append the result to the list
        top_news_sources_list.append(result)

    # Convert the list of results into a DataFrame
    top_news_sources_df = pd.DataFrame(top_news_sources_list)

    return top_news_sources_df

def perform_clustering_news(df, features, n_clusters=5, random_state=42):
    """
    Perform K-means clustering on the provided DataFrame using the specified features.

    Parameters:
    - df: pandas DataFrame containing the data.
    - features: List of column names to be used for clustering.
    - n_clusters: Number of clusters to form.
    - random_state: Random state for reproducibility. Default is 42.

    Returns:
    - df_with_clusters: DataFrame with an additional 'Cluster' column indicating cluster assignments.
    """

    # Encode categorical variables using one-hot encoding
    data_encoded = pd.get_dummies(df[features])

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_encoded)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['Cluster'] = kmeans.fit_predict(data_scaled)

    return df, kmeans, scaler, data_scaled, data_encoded.columns

def find_similar_users_news(new_user_data, features, train_df, kmeans, scaler, data_scaled, data_columns):
    """
    Find the 5 most similar users for a new user based on the trained K-means model.

    Parameters:
    - new_user_data: Dictionary of feature values for the new user.
    - features: List of column names to be used for finding similar users.
    - train_df: DataFrame used to train the K-means model, containing cluster assignments.
    - kmeans: Fitted KMeans model.
    - scaler: Fitted StandardScaler.
    - data_scaled: Scaled training data used for clustering.
    - data_columns: Columns of the encoded data used in clustering.

    Returns:
    - similar_users: List of user IDs for the 5 most similar users.
    """

    # Encode and scale the new user data
    new_user_encoded = pd.get_dummies(pd.DataFrame([new_user_data])[features])
    new_user_encoded = new_user_encoded.reindex(columns=data_columns, fill_value=0)
    new_user_scaled = scaler.transform(new_user_encoded)

    # Predict the cluster for the new user
    new_user_cluster = kmeans.predict(new_user_scaled)

    # Get the indices of users in the same cluster
    cluster_indices = train_df[train_df['Cluster'] == new_user_cluster[0]].index.tolist()

    if not cluster_indices:
        print("No users found in the same cluster.")
        return []

    # Calculate pairwise distances
    distances = pairwise_distances(new_user_scaled, data_scaled[cluster_indices])

    # Get the indices of the n closest users within the same cluster
    closest_indices = distances.argsort()[0][:5]

    # Map these indices back to the original DataFrame indices
    closest_train_indices = [cluster_indices[i] for i in closest_indices]

    # Extract the user IDs of the closest users
    similar_users = train_df.loc[closest_train_indices, 'user_id'].values

    return list(set(similar_users.flatten()))

def rank_top_news_sources_till_three(similar_users_data):
    # Ensure that only numeric columns are used for calculating scores
    numeric_data = similar_users_data.select_dtypes(include=['number'])

    # Sum the scores of the news sources for the similar users
    news_source_scores = numeric_data.sum(axis=0) / len(similar_users_data)

    # Sort the news sources based on the scores in descending order
    sorted_scores = news_source_scores.sort_values(ascending=False)

    # Get the unique scores in descending order
    unique_scores = sorted_scores.unique()

    # Start by getting the top score
    top_score = unique_scores[0]

    # Get the news sources with the top score
    top_news_sources = sorted_scores[sorted_scores == top_score]

    # If we already have 3 or more sources with the top score, return them
    if len(top_news_sources) >= 3:
        return top_news_sources

    # Otherwise, move to the second-highest score
    second_highest_score = unique_scores[1]

    # Get the news sources with the second-highest score
    second_highest_news_sources = sorted_scores[sorted_scores == second_highest_score]

    # Combine the top and second-highest scores
    combined_news_sources = pd.concat([top_news_sources, second_highest_news_sources])

    # If we now have 3 or more sources, return them
    if len(combined_news_sources) >= 3:
        return combined_news_sources

    # If we still have fewer than 3, move to the third-highest score (if it exists)
    if len(unique_scores) > 2:
        third_highest_score = unique_scores[2]
        third_highest_news_sources = sorted_scores[sorted_scores == third_highest_score]

        if len(third_highest_news_sources) > 3:
          return combined_news_sources
        else:
          combined_news_sources = pd.concat([combined_news_sources, third_highest_news_sources])

    return combined_news_sources


'''
END NEWS RANKING
'''


'''
BEGIN RAG SETUP / USAGE
'''
def setup_populate_chromadb(ragdata):
    ids = []
    texts = []
    prepped = 0

    for source, time, text in zip(ragdata["Source"], ragdata["published date"], ragdata["Article Text"]):
        ids.append(str(source + "_" + time + "_" + str(prepped)))
        texts.append(str(text))
        prepped += 1

    # sets up Chroma DB
    chromadb_client = chromadb.HttpClient(port=8000, host=chroma_source)
    collection = chromadb_client.create_collection(name=BIGRAG_name, embedding_function=embedding_functions.ONNXMiniLM_L6_V2()) # works

    added = 0
    distance = 100
    finished = False

    # adds data in reasonable portions of 30,000
    while added != len(ragdata) and finished != True:
        collection.add(
            documents=texts[added:added+distance],
            embeddings=BIGRAG_embeddings[added:added+distance],
            ids=ids[added:added+distance]
        )
        if added == len(ragdata):
            finished = True
        else:
            added += distance
            if added+distance > len(ragdata):
                distance = len(ragdata) - added

    collection.count()
    return collection

def grab_chromadb():
  chromadb_client = chromadb.HttpClient(port=8000, host=chroma_source)
  collection = chromadb_client.get_collection(name=BIGRAG_name, embedding_function=embedding_functions.ONNXMiniLM_L6_V2()) # works
  print(collection.count())
  return collection

def query_chromabd(ragdata, trusted_sources, reasoning):
  excerpts_for_summarization = []

  # queries database for similar results
  similar_articles = ragdata.query(query_texts=[reasoning])

  # goes through each trusted source
  for rated in trusted_sources:
    # goes through each document and finds first most similar response
    for docs, ids in zip(similar_articles["documents"], similar_articles["ids"]):
      for num in range(len(docs)):
        if rated in ids[num] or ids[num] in rated:
          excerpts_for_summarization.append(rated + ":   " + docs[num] + "\n END OF ARTICLE.")
          break

  # when in doubt, return top 2 sources of relevance
  if len(excerpts_for_summarization) == 0:
     print("None found. Returning top 2 results...")
     excerpts_for_summarization.append(similar_articles["ids"][0][0] + ":   " + similar_articles["documents"][0][0] + "\n END OF ARTICLE.")
     excerpts_for_summarization.append(similar_articles["ids"][0][1] + ":   " + similar_articles["documents"][0][1]  + "\n END OF ARTICLE.")

  return excerpts_for_summarization

'''
END RAG SETUP / USAGE
'''

'''
BEGIN RHETORICAL STYLING
'''
def assign_scores(row):
    order = eval(row['ethosPathosLogos'])
    scores = {'Ethos': 0, 'Pathos': 0, 'Logos': 0}

    for i, value in enumerate(order):
        if "Ethos" in value:
            scores['Ethos'] = 3 - i
        elif "Pathos" in value:
            scores['Pathos'] = 3 - i
        elif "Logos" in value:
            scores['Logos'] = 3 - i

    return pd.Series([scores['Ethos'], scores['Pathos'], scores['Logos']], index=['ethos', 'pathos', 'logos'])

def get_train_test_data_persuasion(df, feature_columns, target_columns, test_size=0.2, random_state=42):
    # Split the data into training and test sets
    train_df = df
    test_df = None

    selected_columns = ['user_id'] + feature_columns + target_columns
    train_df = train_df[selected_columns]

    # Extract features based on the passed feature_columns
    train_df_features = train_df[feature_columns]
    test_df_features = None

    # Create user-item matrices
    train_user_item_matrix = pd.concat([train_df.iloc[:, 0], train_df.iloc[:, -3:]], axis=1)
    test_user_item_matrix = None

    return train_df, test_df, train_df_features, test_df_features, train_user_item_matrix, test_user_item_matrix

def perform_clustering_persuasion(df, features, n_clusters=5, random_state=42):
    """
    Perform K-means clustering on the provided DataFrame using the specified features.

    Parameters:
    - df: pandas DataFrame containing the data.
    - features: List of column names to be used for clustering.
    - n_clusters: Number of clusters to form. Default is 5.
    - random_state: Random state for reproducibility. Default is 42.

    Returns:
    - df_with_clusters: DataFrame with an additional 'Cluster' column indicating cluster assignments.
    """

    # Encode categorical variables using one-hot encoding
    data_encoded = pd.get_dummies(df[features])

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_encoded)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['Cluster'] = kmeans.fit_predict(data_scaled)

    return df, kmeans, scaler, data_scaled, data_encoded.columns


def find_similar_users_persuasion(new_user_data, features, train_df, kmeans, scaler, data_scaled, data_columns):
    """
    Find the 5 most similar users for a new user based on the trained K-means model.

    Parameters:
    - new_user_data: Dictionary of feature values for the new user.
    - features: List of column names to be used for finding similar users.
    - train_df: DataFrame used to train the K-means model, containing cluster assignments.
    - kmeans: Fitted KMeans model.
    - scaler: Fitted StandardScaler.
    - data_scaled: Scaled training data used for clustering.
    - data_columns: Columns of the encoded data used in clustering.

    Returns:
    - similar_users: List of user IDs for the 5 most similar users.
    """

    # Encode and scale the new user data
    new_user_encoded = pd.get_dummies(pd.DataFrame([new_user_data])[features])
    new_user_encoded = new_user_encoded.reindex(columns=data_columns, fill_value=0)
    new_user_scaled = scaler.transform(new_user_encoded)

    # Predict the cluster for the new user
    new_user_cluster = kmeans.predict(new_user_scaled)

    # Get the indices of users in the same cluster
    cluster_indices = train_df[train_df['Cluster'] == new_user_cluster[0]].index.tolist()

    if not cluster_indices:
        print("No users found in the same cluster.")
        return []

    # Calculate pairwise distances
    distances = pairwise_distances(new_user_scaled, data_scaled[cluster_indices])

    # Get the indices of the n closest users within the same cluster
    closest_indices = distances.argsort()[0][:5]

    # Map these indices back to the original DataFrame indices
    closest_train_indices = [cluster_indices[i] for i in closest_indices]

    # Extract the user IDs of the closest users
    similar_users = train_df.loc[closest_train_indices, 'user_id'].values

    return similar_users

def rank_persuasion(similar_users_data):
    # Ensure that only numeric columns are used for calculating scores
    numeric_data = similar_users_data.select_dtypes(include=['number'])

    # Sum the scores of the news sources for the similar users
    persuasion_scores = numeric_data.sum(axis=0) / len(similar_users_data)

    # Sort the news sources based on the scores in descending order
    top_persuasion = persuasion_scores.sort_values(ascending=False)

    return top_persuasion

'''
END RHETORICAL STYLING
'''

'''
BEGIN LLM CALLS
'''

def get_summary(misinfo, trusted_sources, content, factuality):
  client = OpenAI(
      api_key="INSERT_KEY_HERE"
  )

  # queries OpenAIi by sending inputs
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "system",
        "content": content
        },
      {
        "role": "user",
        "content": "Information: " + misinfo + ", Excerpts: " + str(trusted_sources)
      }
    ],

    temperature=0.3,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return response.choices[0].message.content

def get_rephrased_persuasion(misinfo, summarized_articles, content):
  client = OpenAI(
      api_key="INSERT_KEY_HERE"
  )
  #queries OpenAI by sending inputs
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "system",
        "content": content
        },
      {
        "role": "user",
        "content": "Information: " + misinfo + ", Summary: " + summarized_articles 
      }
    ],
    temperature=0.3,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return response.choices[0].message.content

'''
END LLM CALLS
'''

'''
FULL PIPELINE CALL
'''

def run_llm_pipeline(user_reasoning, user_profile, round_misinfo, factuality, populated_rag=None):
  # step 0 - put together rag
  if populated_rag == None:
     try:
      populated_rag = setup_populate_chromadb(BIGRAG_full)
     except:
      populated_rag = grab_chromadb()

  # step 1 - source rank

  # Training cluster
  news_targets = ['Washington Examiner', 'The Economist', 'MSNBC News','AP News', 'BBC','Fox News', 'CNN', 'The Washington Post', 'The New York Times', 'New York Post']
  train_df, test_df, train_df_features, test_df_features, train_user_item_matrix, test_user_item_matrix = get_train_test_data_news(dataset1, features, news_targets)
  train_df, kmeans, scaler, data_scaled, data_encoded_columns = perform_clustering_news(train_df, features)
  test_df = pd.DataFrame.from_dict([user_profile])

  # Cluster on the test data by finding similar users for a test instance
  results = []
  news_sources_lengths = []
  for index, row in test_df.iterrows():
      similar_users = find_similar_users_news(row, features, train_df, kmeans, scaler, data_scaled, data_encoded_columns)
      similar_users_data = train_user_item_matrix[train_user_item_matrix['user_id'].isin(similar_users)]
      top_news_sources_for_person = rank_top_news_sources_till_three(similar_users_data)
      news_sources_list_for_person = top_news_sources_for_person.index.tolist()
      news_sources_lengths.append(len(news_sources_list_for_person))
      results.append({'user_id': row['user_id'], 'top_news_sources': news_sources_list_for_person})

  news_rank_results = results
  # step 2 - debunking articles

  if factuality == "true":
     articles_for_summarization = query_chromabd(populated_rag, news_rank_results[0]['top_news_sources'], "Prove the following statement: '" + round_misinfo + "'")
  else:
    articles_for_summarization = query_chromabd(populated_rag, news_rank_results[0]['top_news_sources'], "Disprove the following statement: '" + round_misinfo + "'")

  # step 3 - summarization of articles
  if factuality == "true":
    content = "Given that the 'Information' is true, summarize the relevant facts of each of the 'Excerpts' that support the 'Information'. Keep the source for each of the 'Excerpts' in the summary. After summarizing, remove all summaries not relevant to 'Information'. Do not use any hate speech or vulgar language."
    summarized_articles = get_summary(round_misinfo, articles_for_summarization, content, factuality="true")
  else:
    content = "Given that the 'Information' is false, summarize the relevant facts of each of the 'Excerpts' that disprove the 'Information'. Keep the source for each of the 'Excerpts' in the summary. After summarizing, remove all summaries not relevant to 'Information'. Do not use any hate speech or vulgar language."
    summarized_articles = get_summary(round_misinfo, articles_for_summarization, content, factuality="false")
    
  # step 4 - persuasion rank
  persuasion_targets = ['Ethos', 'Pathos', 'Logos']
  dataset2[['Ethos', 'Pathos', 'Logos']] = dataset2.apply(assign_scores, axis=1)
  dataset2 = dataset2.drop(columns=['ethosPathosLogos'])

  train_df, _, train_df_features, test_df_features, train_user_item_matrix, test_user_item_matrix = get_train_test_data_persuasion(dataset2, features, persuasion_targets)
  train_df = train_df.reset_index(drop=True)

  # Training cluster
  train_df, kmeans, scaler, data_scaled, data_encoded_columns = perform_clustering_persuasion(train_df, features)

  # Cluster on the test data by finding similar users for a test instance
  results = []
  for index, row in test_df.iterrows():
      similar_users = find_similar_users_persuasion(row, features, train_df, kmeans, scaler, data_scaled, data_encoded_columns)
      similar_users_data = train_user_item_matrix[train_user_item_matrix['user_id'].isin(similar_users)]
      top_persuasion_for_person = rank_persuasion(similar_users_data)
      persuasion_list_for_person = top_persuasion_for_person.index.tolist()
      results.append({'user_id': row['user_id'], 'top_persuasion': persuasion_list_for_person})
  top_persuasion_df_predicted = pd.DataFrame(results)
  persuasion_rank_results = top_persuasion_df_predicted

  # step 4.5 - adjust prompt for styling
  if persuasion_rank_results["top_persuasion"][0][0] == "Ethos":
     best_tactic = "emphasizing the credibility of the news sources."
  elif persuasion_rank_results["top_persuasion"][0][0] == "Pathos":
     best_tactic = "emphasizing empathy towards the reader."
  elif persuasion_rank_results["top_persuasion"][0][0] == "Logos":
     best_tactic = "emphasizing the impact of the evidence."
  else:
     best_tactic = "emphasizing the credibility of the news sources."

  # step 5 - style transfer
  if factuality == "true":
    content = "You are an informed citizen, persuading another citizen that the 'Information' is true. Read the 'Summary' and identify relevant facts. Then, write a first person response to prove the 'Information' by " + best_tactic + " Do not use hate speech or vulgar language. Limit the response to 4 sentences."
    style_transferred_message = get_rephrased_persuasion(round_misinfo, summarized_articles, content)
  else:
    content = "You are an informed citizen, persuading another citizen that the 'Information' is false. Read the 'Summary' and identify relevant facts. Then, write a first person response to disprove the 'Information' by " + best_tactic + " Do not use hate speech or vulgar language. Limit the response to 4 sentences."
    style_transferred_message = get_rephrased_persuasion(round_misinfo, summarized_articles, content)

  style_transferred_message = style_transferred_message.replace("Summary: ", "")
  style_transferred_message = style_transferred_message.replace("summary: ", "")
  style_transferred_message = style_transferred_message.replace("Summary:", "")
  style_transferred_message = style_transferred_message.replace("summary:", "")

  if style_transferred_message[0:9] == "Rephrased ":
     style_transferred_message = style_transferred_message[10:]
  if style_transferred_message[0:10] == "Rephrased: ":
     style_transferred_message = style_transferred_message[11:]

  style_transferred_message = style_transferred_message.replace("'Information'", "information")
  style_transferred_message = style_transferred_message.replace("'Information.'", "information.")

  style_transferred_message = style_transferred_message.replace("As an informed citizen,", "")
  style_transferred_message = style_transferred_message.replace("As an informed citizen", "")

  style_transferred_message = style_transferred_message.replace("*", "")

  return style_transferred_message