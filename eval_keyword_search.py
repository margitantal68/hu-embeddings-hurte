import pandas as pd
from rank_bm25 import BM25Okapi
import re
import json

# --------------------
# Cleanservice data - Load topics.txt 
# --------------------
def load_topics(path):
    topics = {}
    current_topic = None
    buffer = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("## "):
                # save previous topic
                if current_topic and buffer:
                    topics[current_topic] = " ".join(buffer).strip()
                    buffer = []
                current_topic = line.replace("## ", "").strip()
            elif line:
                buffer.append(line)
        # save last topic
        if current_topic and buffer:
            topics[current_topic] = " ".join(buffer).strip()
    return topics

# --------------------
# Tokenizer
# --------------------
def tokenize(text):
    return re.findall(r"\w+", text.lower())


# --------------------
# Evaluation - Cleanservice data
# --------------------
def evaluate_cleanservice():
    # --------------------
    # Build BM25 index over topics
    # --------------------
    topics = load_topics("data/cleanservice/topics.txt")
    corpus = list(topics.values())
    topic_names = list(topics.keys())

    corpus_tokens = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(corpus_tokens)

    # --------------------
    # Load Questions CSV
    # --------------------
    questions_df = pd.read_csv("data/cleanservice/cs_qa.csv")  # columns: question, topic, answer

    mrr_total = 0.0
    recall_at_1 = 0
    recall_at_3 = 0
    n = len(questions_df)

    for idx, row in questions_df.iterrows():
        question = row["question"]
        true_topic = row["topic"]

        query_tokens = tokenize(question)
        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Find rank of correct topic
        rank = None
        for r, doc_idx in enumerate(ranked_indices, start=1):
            if topic_names[doc_idx] == true_topic:
                rank = r
                break

        if rank is not None:
            mrr_total += 1.0 / rank
            if rank == 1:
                recall_at_1 += 1
            if rank <= 3:
                recall_at_3 += 1

    mrr = mrr_total / n
    recall1 = recall_at_1 / n
    recall3 = recall_at_3 / n

    return mrr, recall1, recall3


# --------------------
# Evaluation - HURTE data
# --------------------
import json
from rank_bm25 import BM25Okapi

def evaluate_hurte_bm25(filename):
    # Load data
    with open(filename, 'r', encoding='utf-8') as f:
        rte_data = json.load(f)
    print(f"RTE data loaded: {len(rte_data)} items.")

    # Keep only positive label items
    rte_data = [item for item in rte_data if str(item.get("label", "")) == "1"]
    print(f"Positive RTE data loaded: {len(rte_data)} items.")

    # Prepare corpus (premises)
    premises = [item["premise"] for item in rte_data]
    tokenized_premises = [premise.split() for premise in premises]

    # Initialize BM25 index
    bm25 = BM25Okapi(tokenized_premises)

    # Prepare queries (hypotheses)
    hypotheses = [item["hypothesis"] for item in rte_data]
    tokenized_hypotheses = [hyp.split() for hyp in hypotheses]

    # Evaluation metrics
    k = 3
    recall_at_1 = 0
    recall_at_3 = 0
    reciprocal_ranks = []
    num_queries = len(hypotheses)

    for i, query in enumerate(tokenized_hypotheses):
        scores = bm25.get_scores(query)
        ranked_indices = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        top_k = ranked_indices[:k]

        # Check Recall@1
        if i == top_k[0]:
            recall_at_1 += 1

        # Check Recall@3
        if i in top_k:
            recall_at_3 += 1

        # Compute reciprocal rank
        if i in ranked_indices:
            rank = ranked_indices.index(i) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    mrr = sum(reciprocal_ranks) / num_queries
    recall_at_1_score = recall_at_1 / num_queries
    recall_at_3_score = recall_at_3 / num_queries

    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Recall@1: {recall_at_1_score:.4f}")
    print(f"Recall@3: {recall_at_3_score:.4f}")


if __name__ == "__main__":
    # mrr, r1, r3 = evaluate_cleanservice()
    # print("## Cleanservice")
    # print(f"MRR: {mrr:.3f}")
    # print(f"Recall@1: {r1:.3f}")
    # print(f"Recall@3: {r3:.3f}")
    # print("## HURTE: ")
    # evaluate_hurte_bm25('data/hurte/rte_dev.json')
    evaluate_hurte_bm25('data/hurte/rte_train.json')
   


