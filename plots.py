import matplotlib.pyplot as plt
import pandas as pd

def plot_model_comparison():
    # Data
    data = {
        "Model": [
            "PP-ML-MINILM",
            "OPENAI-ADA",
            "OPENAI-3 SMALL",
            "NOMIC",
            "MINILM",
            "GEMINI",
            "BGE-M3"
        ],
        "MRR": [0.84, 0.80, 0.80, 0.71, 0.59, 0.50, 0.90],
        "Recall@1": [0.78, 0.72, 0.72, 0.64, 0.46, 0.38, 0.86],
        "Recall@3": [0.92, 0.90, 0.94, 0.80, 0.74, 0.68, 0.96]
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

# Call the function
# plot_model_comparison()


import matplotlib.pyplot as plt

def plot_recall_at_3(data):
    models = data["Model"]
    recall_values = data["Recall@3"]

    # Sort the data by Recall@3 descending
    sorted_data = sorted(zip(models, recall_values), key=lambda x: x[1], reverse=True)
    sorted_models, sorted_recalls = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_models, sorted_recalls, color='skyblue')
    plt.xlabel("Recall@3")
    plt.title("Model Comparison by Recall@3")
    plt.xlim(0, 1.05)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}', va='center')

    plt.gca().invert_yaxis()  # Highest recall on top
    plt.tight_layout()
    plt.show()

# Example usage:
data = {
    "Model": [
        "PP-ML-MINILM",
        "OPENAI-ADA",
        "OPENAI-3 SMALL",
        "NOMIC",
        "MINILM",
        "GEMINI",
        "BGE-M3"
    ],
    "Recall@3": [0.92, 0.90, 0.94, 0.80, 0.74, 0.68, 0.96]
}

plot_recall_at_3(data)
