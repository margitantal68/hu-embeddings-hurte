import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_model_comparison():
    # Data
    data = {
        "Model": [
            "PP-MINILM",
            "OPENAI-ADA",
            "OPENAI-3SMALL",
            "NOMIC",
            "MINILM",
            "GEMINI",
            "BGE-M3",
            "HUBERT"
        ],
        "MRR": [0.84, 0.80, 0.80, 0.71, 0.59, 0.50, 0.90, 0.48],
        "Recall@1": [0.78, 0.72, 0.72, 0.64, 0.46, 0.38, 0.86, 0.38],
        "Recall@3": [0.92, 0.90, 0.94, 0.80, 0.74, 0.68, 0.96, 0.60]
    }

    df = pd.DataFrame(data)
    bar_width = 0.2
    x = range(len(df))

    # Bar positions
    positions_mrr = [i - bar_width for i in x]
    positions_r1 = x
    positions_r3 = [i + bar_width for i in x]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(positions_mrr, df["MRR"], width=bar_width, label="MRR", color="#1f77b4")
    ax.bar(positions_r1, df["Recall@1"], width=bar_width, label="Recall@1", color="#ff7f0e")
    ax.bar(positions_r3, df["Recall@3"], width=bar_width, label="Recall@3", color="#2ca02c")

    # Labels and titles
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: MRR, Recall@1, Recall@3")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()



def plot_recall_at_3_clearservice():
    data = {
        "Model": [
            "BGE-M3",
                "GEMINI",
                "HUBERT",
                "MINILM",
                "NOMIC",
                "OPENAI-3SMALL",
                "OPENAI-ADA",
                "PP-MINILM",
                "BM25"
            
        ],
        "Recall@3": [0.96, 0.68, 0.60, 0.74, 0.80, 0.94, 0.90, 0.92, 0.80] # Cleanservice
    }

    models = data["Model"]
    recall_values = data["Recall@3"]

    # Sort the data by Recall@3 descending
    sorted_data = sorted(zip(models, recall_values), key=lambda x: x[1], reverse=True)
    sorted_models, sorted_recalls = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_models, sorted_recalls, color='skyblue')
    plt.xlabel("Recall@3")
    plt.title("Clearservice dataset")
    
    plt.xlim(0, 1.05)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}', va='center')

    plt.gca().invert_yaxis()  # Highest recall on top
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_recall_at_3_hurte(data=None):
    # Use provided data or the default HuRTE values
    if data is None:
        data = {
            "Model": [
                "BGE-M3",
                "GEMINI",
                "HUBERT",
                "MINILM",
                "NOMIC",
                "OPENAI-3SMALL",
                "OPENAI-ADA",
                "PP-MINILM",
                "BM25"
            ],
            "training":    [0.97, 0.77, 0.66, 0.61, 0.80, 0.92, 0.92, 0.91, 0.79],  # HuRTE - training
            "development": [1.00, 0.86, 0.86, 0.77, 0.95, 0.97, 0.98, 0.96, 0.84]   # HuRTE - development
        }


    models = list(data["Model"])
    training_values = list(data["training"])
    dev_values = list(data["development"])

    # Keep the original order — no sorting
    y = np.arange(len(models))
    bar_h = 0.38

    plt.figure(figsize=(10, 6))
    bars_train = plt.barh(y - bar_h/2, training_values, height=bar_h, label="train", color="skyblue")
    bars_dev   = plt.barh(y + bar_h/2, dev_values,    height=bar_h, label="dev", color="lightgreen")

    plt.yticks(y, models)
    plt.xlabel("Recall@3")
    plt.title("HuRTE dataset: Training vs Development")
    plt.xlim(0, 1.05)
    plt.legend()

    # Annotate bar values
    for bars in (bars_train, bars_dev):
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va="center")

    # Put the first listed model at the top
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


# plot_recall_at_3_clearservice()
# plot_recall_at_3_hurte()
plot_model_comparison()


