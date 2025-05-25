import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import argparse
import json
import os

def visualize_embedding_distribution(embedding_path, subset_path, output_path=None, label='Subset'):
    print(f"Loading full embeddings from: {embedding_path}")
    all_embeddings = np.load(embedding_path)

    print(f"Loading subset indices from: {subset_path}")
    if subset_path.endswith(".json"):
        with open(subset_path, 'r') as f:
            selected_indices = json.load(f)
        selected_indices = np.array(selected_indices, dtype=int)
    else:
        df = pd.read_parquet(subset_path)
        print(f"Loaded Parquet File Shape: {df.shape}")
        print("First few rows of the file:")
        print(df.head())

        selected_indices = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        selected_indices = selected_indices.dropna().astype(int).values

    print("Fitting UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)

    labels = np.full(all_embeddings.shape[0], 'All', dtype=object)
    labels[selected_indices] = label

    print("Plotting...")
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding_2d[labels == 'All', 0], embedding_2d[labels == 'All', 1],
                alpha=0.3, label='All Data', s=5)
    plt.scatter(embedding_2d[labels == label, 0], embedding_2d[labels == label, 1],
                alpha=0.8, label=label, s=8, c='red')
    plt.legend()
    plt.title(f'UMAP Projection of All vs {label} Embeddings')

    if output_path:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, default="embeddings_checkpoint_alpaca.npy")
    parser.add_argument("--subset_path", type=str, required=True)
    parser.add_argument("--label", type=str, default="Subset")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    visualize_embedding_distribution(
        embedding_path=args.embedding_path,
        subset_path=args.subset_path,
        output_path=args.output_path,
        label=args.label
    )


'''
LD_LIBRARY_PATH=/home/selen/anaconda3/envs/algo/lib:$LD_LIBRARY_PATH \
python visualize_umap.py \
  --embedding_path embeddings_checkpoint_alpaca.npy \
  --subset_path recovered_indices/craig_selected_10352_indices.json \
  --label "CRAIG 20%" \
  --output_path umap_plot/craig_umap_20.png

'''


