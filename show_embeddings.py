import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Optional: if you have UMAP installed
try:
    import umap
    USE_UMAP = True
except ImportError:
    USE_UMAP = False

# Import your models from models.py
from models import (
    OllamaEmbedder,
    SentenceTransformer,
    OpenAIEmbedder,
    GeminiEmbedder,
    BGEEmbedder,
    SentenceTransformerEmbedder
)

# 1. Prepare models
models = [
    OllamaEmbedder("nomic-embed-text:latest"),
    OllamaEmbedder("all-minilm:latest"),
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
    OpenAIEmbedder("text-embedding-ada-002"),
    OpenAIEmbedder("text-embedding-3-small"),
    GeminiEmbedder(),
    BGEEmbedder(),
    SentenceTransformerEmbedder()
]

# 2. Load topics from file
def load_topics(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Split on '##', remove empty entries, strip spaces
    topics = [t.strip() for t in re.split(r'^##', text, flags=re.MULTILINE) if t.strip()]
    return topics

topics = load_topics("data/cleanservice/topics.txt")
print(f"Loaded {len(topics)} topics.")

# 3. Visualization and evaluation function
def visualize_embeddings(model_name, model, topics, method="tsne"):
    # Get model name for labeling
    # model_name = type(model).__name__
    try:
        name_attr = getattr(model, 'model_name', None)
        if name_attr:
            model_name += f" ({name_attr})"
    except:
        pass

    print(f"\nProcessing model: {model_name}")

    # Encode topics
    embeddings = model.encode(topics)
    embeddings = np.array(embeddings)

    # Standardize before dimensionality reduction
    embeddings_scaled = StandardScaler().fit_transform(embeddings)

    # Dimensionality reduction
    if method == "umap" and USE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(topics) - 1))
    reduced = reducer.fit_transform(embeddings_scaled)



    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], s=50)
    plt.title(f"{model_name} - {method.upper()} projection")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
   
    plt.show()
    plt.savefig('figures/' + model_name + '.png')

    # Clustering quality
    # if len(set(topics)) > 1:
    #     sil_score = silhouette_score(embeddings_scaled, list(range(len(topics))))
    #     db_score = davies_bouldin_score(embeddings_scaled, list(range(len(topics))))
    # else:
    #     sil_score, db_score = np.nan, np.nan

    # print(f"Silhouette Score: {sil_score:.4f}")
    # print(f"Davies–Bouldin Score: {db_score:.4f}")

# 4. Run for all models
# for m in models:
#     visualize_embeddings(m, topics, method="umap" if USE_UMAP else "tsne")


model_names = [
    "nomic-embed-text:latest",
    "all-minilm:latest",
    "paraphrase-multilingual-MiniLM-L12-v2'",
    "OpenAI - text-embedding-ada-002",
    "OpenAI - text-embedding-3-small",
    "Gemini - embedding-001",
    "BGE-M3",
    "HuBERT"
]
if __name__ == "__main__":
    index = 7
    model_name = model_names[index]
    model = models[index]  
    visualize_embeddings(model_name, model, topics, method="umap" if USE_UMAP else "tsne")