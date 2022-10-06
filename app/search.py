# Importing packages

import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import json
import time
import os

import warnings
warnings.filterwarnings('ignore')


SEARCH_FILES_DIR = 'app/search_files'

df = pd.read_csv(os.path.join(SEARCH_FILES_DIR, 'walmart_data/walmart_product_data_30k.csv'))


st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

with open(os.path.join(SEARCH_FILES_DIR, 'sentence_transformer/category_clustered_full_text_all_vectors.pkl'), 'rb') as f:
    category_clustered_full_text_all_vectors = pickle.load(f)

with open(os.path.join(SEARCH_FILES_DIR, 'sentence_transformer/full_text_vectors.pkl'), 'rb') as f:
    full_text_vectors = pickle.load(f)

with open(os.path.join(SEARCH_FILES_DIR, 'sentence_transformer/category_clustered_full_text_idx.pkl'), 'rb') as f:
    category_clustered_full_text_idx = pickle.load(f)
    
with open(os.path.join(SEARCH_FILES_DIR, 'sentence_transformer/category_clustered_full_text_mean_vectors.pkl'), 'rb') as f:
    category_clustered_full_text_mean_vectors = pickle.load(f)
    
with open(os.path.join(SEARCH_FILES_DIR, 'sentence_transformer/vectorized_corpus.pkl'), 'rb') as f:
    vectorized_corpus = pickle.load(f)
    
with open(os.path.join(SEARCH_FILES_DIR, 'sentence_transformer/category_keys_list.pkl'), 'rb') as f:
    category_keys_list = pickle.load(f)


def sent2embed(sent, st_model, cleaned=False):
    # clean text
    if not cleaned:
        sent = re.sub('[^a-zA-Z]', ' ', sent)
        sent = ' '.join([word.lower() for word in sent.split()]).strip()
    return st_model.encode(sent, show_progress_bar=False)


def get_top_n(vec_query, vec_corpus, n=10):
    scores = []
    for v in vec_corpus:
        scores.append(cosine_similarity(vec_query.reshape(1, -1), v.reshape(1, -1))[0][0])
    return np.argsort(np.array(scores))[-n:][::-1]


def get_results(query, model=st_model, return_n=10, top_n_products=50, top_n_categories=3):
    st = time.time()
    # convert given query into 300d vector
    # vectorized_query = sent2vec(query, w2v_model)
    vectorized_query = sent2embed(query, model)
    
    # get index of top n matching categories
    top_cat_idx = get_top_n(vectorized_query, vectorized_corpus, n=top_n_categories)
    # print(np.array(category_keys_list)[top_cat_idx])
    
    cat0 = category_keys_list[top_cat_idx[0]]
    cat1 = category_keys_list[top_cat_idx[1]]
    cat2 = category_keys_list[top_cat_idx[2]]
    
    vec_corpus0 = category_clustered_full_text_all_vectors[cat0]
    vec_corpus1 = category_clustered_full_text_all_vectors[cat1]
    vec_corpus2 = category_clustered_full_text_all_vectors[cat2]
    
    # get index of top 10 vectorized full text matches from the selected top category's vectorized corpus
    top_full_text_all_vectors_idx0 = get_top_n(vectorized_query, vec_corpus0, n=top_n_products)
    top_full_text_all_vectors_idx1 = get_top_n(vectorized_query, vec_corpus1, n=top_n_products)
    top_full_text_all_vectors_idx2 = get_top_n(vectorized_query, vec_corpus2, n=top_n_products)
    
    # get indexes of top product matches
    top_product_indexes = np.array(category_clustered_full_text_idx[cat0])[top_full_text_all_vectors_idx0].tolist() + \
                          np.array(category_clustered_full_text_idx[cat1])[top_full_text_all_vectors_idx1].tolist() + \
                          np.array(category_clustered_full_text_idx[cat2])[top_full_text_all_vectors_idx2].tolist()
    
    time_req = round(time.time() - st, 4)

    return time_req, json.loads(df.iloc[top_product_indexes, [0, 1, 2, 3, 5, 6]].to_json(orient='records'))[:return_n]
