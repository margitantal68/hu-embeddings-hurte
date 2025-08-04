import os
import re
import faiss
import json

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


from models import BGEEmbedder, OllamaEmbedder, OpenAIEmbedder, GeminiEmbedder 



    
def build_faiss_index(texts, model):
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    print('SHAPE: ', embeddings.shape)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, embeddings


# RTE data
def evaluate_models_rte_data(filename, model):
    # Load data
    with open(filename, 'r', encoding='utf-8') as f:
        rte_data = json.load(f)
    print(f"RTE data loaded: {len(rte_data)} items.")
  
    # Prepare data for embedding
    rte_data = [item for item in rte_data if str(item.get("label", "")) == "1"]
    print(f"Positive RTE data loaded: {len(rte_data)} items.")

    data_to_embed = []
    for item in rte_data:
        premise = item['premise']
        data_to_embed.append(f"{premise}")

    # Create FAISS index
    index, embeddings = build_faiss_index(data_to_embed, model)

    # Evaluate retriever with the embedder model
    data_to_search = []
    for item in rte_data:
        hypothesis = item['hypothesis']
        data_to_search.append(f"{hypothesis}")

    query_vecs = model.encode(data_to_search, convert_to_numpy=True, normalize_embeddings=True)
    k = 3
    recall_at_1 = 0
    recall_at_3 = 0
    reciprocal_ranks = []
    num_queries = len(query_vecs)

    for i, query_vec in enumerate(query_vecs):
        D, I = index.search(query_vec.reshape(1, -1), k)
        # The correct premise is at index i
        retrieved_indices = I[0]
        if i == retrieved_indices[0]:
            recall_at_1 += 1
        if i in retrieved_indices:
            recall_at_3 += 1
        if i in retrieved_indices:
            rank = retrieved_indices.tolist().index(i) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    mrr = sum(reciprocal_ranks) / num_queries
    recall_at_1_score = recall_at_1 / num_queries
    recall_at_3_score = recall_at_3 / num_queries

    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Recall@1: {recall_at_1_score:.4f}")
    print(f"Recall@3: {recall_at_3_score:.4f}")

# Cleanservice data 

def parse_topics(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    
    topics = []
    chunks = re.split(r"^##\s*", content, flags=re.MULTILINE)
    for chunk in chunks[1:]:  # első elem üres lehet
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        description = "\n".join(lines[1:]).strip()
        text = f"{title}: {description}"
        topics.append(text)
    return topics

def evaluate_models_cleanservice_data(model):
    # Load questions
    df = pd.read_csv("data/cleanservice/cs_qa.csv")
    print(f"Cleanservice data loaded: {len(df)} items.")
    

    # Create FAISS index
    topic_chunks = parse_topics("data/cleanservice/topics.txt")
    index, embeddings = build_faiss_index(topic_chunks, model)

    # Evaluate retriever with the embedder model
    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_3 = 0
    num_questions = len(df)
    for idx, question in enumerate(df['question']):
        topic = df['topic'][idx]
        query_vec = model.encode([question], normalize_embeddings=True)
        D, I = index.search(query_vec, k=3)
        top_texts = [topic_chunks[i] for i in I[0]]
        result = [text.split(':', 1)[0].strip() for text in top_texts]
        # Recall@1
        if topic == result[0]:
            recall_at_1 += 1
        # Recall@3
        if topic in result:
            recall_at_3 += 1
            rank = result.index(topic) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    if num_questions > 0:
        recall_at_1_score = recall_at_1 / num_questions
        recall_at_3_score = recall_at_3 / num_questions
        mrr = sum(reciprocal_ranks) / num_questions
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f"Recall@1: {recall_at_1_score:.4f}")
        print(f"Recall@3: {recall_at_3_score:.4f}")
        
    else:
        print("No questions to evaluate.")


models = [ OllamaEmbedder("nomic-embed-text:latest"),
           OllamaEmbedder("all-minilm:latest"),
           SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
           OpenAIEmbedder("text-embedding-ada-002"),
           OpenAIEmbedder("text-embedding-3-small"),
           GeminiEmbedder(),
           BGEEmbedder() ]


if __name__ == "__main__":
    model = models[5]  
    # evaluate_models_rte_data("data/hurte/rte_dev.json", model)
    evaluate_models_cleanservice_data(model)
    print("Evaluation completed.")
    

    


    
    