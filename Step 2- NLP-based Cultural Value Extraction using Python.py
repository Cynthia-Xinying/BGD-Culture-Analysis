

# -------------------------- Step 2: NLP-based Cultural Value Extraction using Python --------------------------
# This step refines cultural value extraction using NLP techniques from Chen (2024).
"""

pip install pandas numpy gensim scikit-learn
import pandas as pd
import numpy as np
import re
import os
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Define an extended list of stopwords
stopwords = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 're', 've', 'y'
])

# Preprocessing text function
def preprocess_text(text):
    if pd.isnull(text):
        return [] 
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = text.split()  # Tokenize the text
    tokens = [word for word in tokens if word not in stopwords]  # Remove stopwords
    return tokens

# Load the dataset
glassdoor_reviews = pd.read_csv('/Applications/Stata/generated Glassdor/generated/generated_bd_20240724_054600_0_1.csv', low_memory=False)

# Preprocess the pros and cons reviews
glassdoor_reviews['processed_pros'] = glassdoor_reviews['review_pros'].apply(preprocess_text)
glassdoor_reviews['processed_cons'] = glassdoor_reviews['review_cons'].apply(preprocess_text)

# Train separate Word2Vec models for pros and con
pros_reviews = glassdoor_reviews['processed_pros'].tolist()
cons_reviews = glassdoor_reviews['processed_cons'].tolist()

# Train Word2Vec models
model_pros = Word2Vec(sentences=pros_reviews, vector_size=100, window=5, min_count=5, workers=4)
model_cons = Word2Vec(sentences=cons_reviews, vector_size=100, window=5, min_count=5, workers=4)

# Seed words for each dimension from the provided table
seed_words = {
    'Innovation': [
        'innovation', 'innovate', 'innovative', 'creativity', 'creative', 'create',
        'passion', 'passionate', 'efficiency', 'efficient', 'excellent', 'pride'
    ],
    'Integrity': [
        'integrity', 'ethic', 'ethical', 'accountable', 'accountability', 'trust',
        'honesty', 'honest', 'honestly', 'fairness', 'responsibility', 'responsible',
        'transparency', 'transparent'
    ],
    'Quality': [
        'quality', 'customer', 'customer_commitment', 'dedication', 'dedicated',
        'dedicate', 'customer_expectation'
    ],
    'Respect': [
        'respectful', 'talent', 'talented', 'employee', 'dignity', 'empowerment', 'empower'
    ],
    'Teamwork': [
        'teamwork', 'collaboration', 'collaborate', 'collaborative', 'cooperation',
        'cooperate', 'cooperative'
    ]
}


# Calculate top similar words for pros and cons
def get_top_similar_words(seed_words, model, top_n=500):
    vocab = list(model.wv.index_to_key)
    similar_words = {}

    # Get seed word vectors
    seed_vectors = [model.wv[word] for word in seed_words if word in model.wv]

    for word in vocab:
        if word not in seed_words:
            word_vector = model.wv[word].reshape(1, -1)
            similarity_scores = [cosine_similarity(word_vector, seed_vector.reshape(1, -1))[0][0] for seed_vector in seed_vectors]
            avg_similarity = np.mean(similarity_scores)
            similar_words[word] = avg_similarity

    # Sort by similarity and return top_n words
    sorted_words = sorted(similar_words.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_n]]

# Get top 500 words for each cultural dimension based on seed words for pros and cons
top_500_words_per_dimension_pros = {}
top_500_words_per_dimension_cons = {}

for dimension, seed_list in seed_words.items():
    top_500_words_per_dimension_pros[dimension] = get_top_similar_words(seed_list, model_pros)
    top_500_words_per_dimension_cons[dimension] = get_top_similar_words(seed_list, model_cons)

# Select top 30 words for each cultural dimension from the top 500
top_30_words_per_dimension_pros = {dim: words[:30] for dim, words in top_500_words_per_dimension_pros.items()}
top_30_words_per_dimension_cons = {dim: words[:30] for dim, words in top_500_words_per_dimension_cons.items()}

# Save the results to CSV files
pros_output_df = pd.DataFrame.from_dict(top_30_words_per_dimension_pros, orient='index').transpose()
cons_output_df = pd.DataFrame.from_dict(top_30_words_per_dimension_cons, orient='index').transpose()

output_path = '/Applications/Stata/generated Glassdor/'
pros_output_df.to_csv(os.path.join(output_path,'top_30_words_pros_0_1.csv'), index=False)
cons_output_df.to_csv(os.path.join(output_path,'top_30_words_cons_0_1.csv'), index=False)

# Calculate weighted scores for each cultural dimension for pros and cons
weighted_scores_pros = {}
weighted_scores_cons = {}

for dimension, seed_list in seed_words.items():
    # Get seed word vectors
    seed_vectors_pros = [model_pros.wv[word] for word in seed_list if word in model_pros.wv]
    seed_vectors_cons = [model_cons.wv[word] for word in seed_list if word in model_cons.wv]
    
    # Calculate weighted score for pros
    total_similarity_score_pros = 0
    count_pros = 0
    for seed_vector in seed_vectors_pros:
        for word in model_pros.wv.index_to_key:
            word_vector = model_pros.wv[word].reshape(1, -1)
            similarity_score = cosine_similarity(word_vector, seed_vector.reshape(1, -1))[0][0]
            total_similarity_score_pros += similarity_score
            count_pros += 1
    weighted_scores_pros[dimension] = total_similarity_score_pros / count_pros if count_pros > 0 else 0

    # Calculate weighted score for cons
    total_similarity_score_cons = 0
    count_cons = 0
    for seed_vector in seed_vectors_cons:
        for word in model_cons.wv.index_to_key:
            word_vector = model_cons.wv[word].reshape(1, -1)
            similarity_score = cosine_similarity(word_vector, seed_vector.reshape(1, -1))[0][0]
            total_similarity_score_cons += similarity_score
            count_cons += 1
    weighted_scores_cons[dimension] = total_similarity_score_cons / count_cons if count_cons > 0 else 0

# Sum the five organizational cultural scores to create aggregate cultural scores
aggregate_cultural_score_pros = sum(weighted_scores_pros.values())
aggregate_cultural_score_cons = sum(weighted_scores_cons.values())

# Save the weighted scores to CSV
weighted_scores_df = pd.DataFrame({
    'Culture Dimension': list(weighted_scores_pros.keys()),
    'Weighted Score Pros': list(weighted_scores_pros.values()),
    'Weighted Score Cons': list(weighted_scores_cons.values())
})

aggregate_scores_df = pd.DataFrame({
    'Aggregate Score Type': ['Pros', 'Cons'],
    'Aggregate Score': [aggregate_cultural_score_pros, aggregate_cultural_score_cons]
})

# Save to specified directory
output_path = '/Applications/Stata/generated Glassdor/'
weighted_scores_df.to_csv(os.path.join(output_path, 'weighted_scores_cultural_dimensions_0_1.csv'), index=False)
aggregate_scores_df.to_csv(os.path.join(output_path, 'aggregate_cultural_scores_0_1.csv'), index=False)





# Define an extended list of stopwords
stopwords = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 're', 've', 'y'
])

# Preprocessing text function
def preprocess_text(text):
    if pd.isnull(text):
        return [] 
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = text.split()  # Tokenize the text
    tokens = [word for word in tokens if word not in stopwords]  # Remove stopwords
    return tokens

# Load the dataset
glassdoor_reviews = pd.read_csv('/Applications/Stata/generated Glassdor/generated/generated_bd_20240724_054600_0_0.csv', low_memory=False)

# Preprocess the pros and cons reviews
glassdoor_reviews['processed_pros'] = glassdoor_reviews['review_pros'].apply(preprocess_text)
glassdoor_reviews['processed_cons'] = glassdoor_reviews['review_cons'].apply(preprocess_text)

# Train separate Word2Vec models for pros and con
pros_reviews = glassdoor_reviews['processed_pros'].tolist()
cons_reviews = glassdoor_reviews['processed_cons'].tolist()

# Train Word2Vec models
model_pros = Word2Vec(sentences=pros_reviews, vector_size=100, window=5, min_count=5, workers=4)
model_cons = Word2Vec(sentences=cons_reviews, vector_size=100, window=5, min_count=5, workers=4)

# Seed words for each dimension from the provided table
seed_words = {
    'Innovation': [
        'innovation', 'innovate', 'innovative', 'creativity', 'creative', 'create',
        'passion', 'passionate', 'efficiency', 'efficient', 'excellent', 'pride'
    ],
    'Integrity': [
        'integrity', 'ethic', 'ethical', 'accountable', 'accountability', 'trust',
        'honesty', 'honest', 'honestly', 'fairness', 'responsibility', 'responsible',
        'transparency', 'transparent'
    ],
    'Quality': [
        'quality', 'customer', 'customer_commitment', 'dedication', 'dedicated',
        'dedicate', 'customer_expectation'
    ],
    'Respect': [
        'respectful', 'talent', 'talented', 'employee', 'dignity', 'empowerment', 'empower'
    ],
    'Teamwork': [
        'teamwork', 'collaboration', 'collaborate', 'collaborative', 'cooperation',
        'cooperate', 'cooperative'
    ]
}


# Calculate top similar words for pros and cons
def get_top_similar_words(seed_words, model, top_n=500):
    vocab = list(model.wv.index_to_key)
    similar_words = {}

    # Get seed word vectors
    seed_vectors = [model.wv[word] for word in seed_words if word in model.wv]

    for word in vocab:
        if word not in seed_words:
            word_vector = model.wv[word].reshape(1, -1)
            similarity_scores = [cosine_similarity(word_vector, seed_vector.reshape(1, -1))[0][0] for seed_vector in seed_vectors]
            avg_similarity = np.mean(similarity_scores)
            similar_words[word] = avg_similarity

    # Sort by similarity and return top_n words
    sorted_words = sorted(similar_words.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_n]]

# Get top 500 words for each cultural dimension based on seed words for pros and cons
top_500_words_per_dimension_pros = {}
top_500_words_per_dimension_cons = {}

for dimension, seed_list in seed_words.items():
    top_500_words_per_dimension_pros[dimension] = get_top_similar_words(seed_list, model_pros)
    top_500_words_per_dimension_cons[dimension] = get_top_similar_words(seed_list, model_cons)

# Select top 30 words for each cultural dimension from the top 500
top_30_words_per_dimension_pros = {dim: words[:30] for dim, words in top_500_words_per_dimension_pros.items()}
top_30_words_per_dimension_cons = {dim: words[:30] for dim, words in top_500_words_per_dimension_cons.items()}

# Save the results to CSV files
pros_output_df = pd.DataFrame.from_dict(top_30_words_per_dimension_pros, orient='index').transpose()
cons_output_df = pd.DataFrame.from_dict(top_30_words_per_dimension_cons, orient='index').transpose()

output_path = '/Applications/Stata/generated Glassdor/'
pros_output_df.to_csv(os.path.join(output_path,'top_30_words_pros_0_0.csv'), index=False)
cons_output_df.to_csv(os.path.join(output_path,'top_30_words_cons_0_0.csv'), index=False)

# Calculate weighted scores for each cultural dimension for pros and cons
weighted_scores_pros = {}
weighted_scores_cons = {}

for dimension, seed_list in seed_words.items():
    # Get seed word vectors
    seed_vectors_pros = [model_pros.wv[word] for word in seed_list if word in model_pros.wv]
    seed_vectors_cons = [model_cons.wv[word] for word in seed_list if word in model_cons.wv]
    
    # Calculate weighted score for pros
    total_similarity_score_pros = 0
    count_pros = 0
    for seed_vector in seed_vectors_pros:
        for word in model_pros.wv.index_to_key:
            word_vector = model_pros.wv[word].reshape(1, -1)
            similarity_score = cosine_similarity(word_vector, seed_vector.reshape(1, -1))[0][0]
            total_similarity_score_pros += similarity_score
            count_pros += 1
    weighted_scores_pros[dimension] = total_similarity_score_pros / count_pros if count_pros > 0 else 0

    # Calculate weighted score for cons
    total_similarity_score_cons = 0
    count_cons = 0
    for seed_vector in seed_vectors_cons:
        for word in model_cons.wv.index_to_key:
            word_vector = model_cons.wv[word].reshape(1, -1)
            similarity_score = cosine_similarity(word_vector, seed_vector.reshape(1, -1))[0][0]
            total_similarity_score_cons += similarity_score
            count_cons += 1
    weighted_scores_cons[dimension] = total_similarity_score_cons / count_cons if count_cons > 0 else 0

# Sum the five organizational cultural scores to create aggregate cultural scores
aggregate_cultural_score_pros = sum(weighted_scores_pros.values())
aggregate_cultural_score_cons = sum(weighted_scores_cons.values())

# Save the weighted scores to CSV
weighted_scores_df = pd.DataFrame({
    'Culture Dimension': list(weighted_scores_pros.keys()),
    'Weighted Score Pros': list(weighted_scores_pros.values()),
    'Weighted Score Cons': list(weighted_scores_cons.values())
})

aggregate_scores_df = pd.DataFrame({
    'Aggregate Score Type': ['Pros', 'Cons'],
    'Aggregate Score': [aggregate_cultural_score_pros, aggregate_cultural_score_cons]
})

# Save to specified directory
output_path = '/Applications/Stata/generated Glassdor/'
weighted_scores_df.to_csv(os.path.join(output_path, 'weighted_scores_cultural_dimensions_0_0.csv'), index=False)
aggregate_scores_df.to_csv(os.path.join(output_path, 'aggregate_cultural_scores_0_0.csv'), index=False)



# -------------------------- Step 3: Fuzzy Matching --------------------------
# This step standardizes company names and performs fuzzy matching to align Glassdoor data with the sample dataset

from rapidfuzz import process, fuzz

def clean_company_name(name):
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())  # Remove non-alphanumeric characters
    name = re.sub(r"\b(inc|corp|ltd|llc|group|company)\b", "", name).strip()
    return name

# Load company datasets
df_sample = pd.read_stata("boardex_sample.dta")
df_generated = pd.read_csv("generated_glassdoor_data.csv")

df_sample["company_clean"] = df_sample["companyname"].apply(clean_company_name)
df_generated["company_clean"] = df_generated["company_name"].apply(clean_company_name)

# Perform fuzzy matching
def fuzzy_match(company, choices, threshold=75):
    match = process.extractOne(company, choices, scorer=fuzz.token_sort_ratio)
    return match[0] if match and match[1] >= threshold else None

sample_names = df_sample["company_clean"].tolist()
df_generated["matched_company"] = df_generated["company_clean"].apply(lambda x: fuzzy_match(x, sample_names))

# Merge datasets
df_matched = df_generated.merge(df_sample, left_on="matched_company", right_on="company_clean", suffixes=("_gen", "_sample"))
df_matched.to_csv("matched_glassdoor_boardex.csv", index=False)

# -------------------------- Summary --------------------------
# This script follows a structured approach:
# 1. Uses Stata to generate initial cultural value scores based on keyword frequency.
# 2. Applies NLP techniques with Word2Vec to refine cultural dimension extraction.
# 3. Performs fuzzy matching to align with sample datasets for better analysis.

# The final dataset, 'matched_glassdoor_boardex.csv', can be used to test the impact of BGD on corporate culture.
