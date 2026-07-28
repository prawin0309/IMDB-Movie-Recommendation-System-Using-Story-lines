"""Render static figures for the README and the project report.

Corpus profile, TF-IDF vocabulary, similarity distribution and topic clusters.
Run after the pipeline and models:

    python data_pipeline.py
    python models.py
    python make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import models  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#2E5E8A", "#C1666B", "#4E9F6E", "#D4A24C", "#7C6A9B", "#5B8C93"]
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"[fig] {name}")


def main() -> None:
    frame = pd.read_csv(config.CLEANED_CSV)

    # 1. Storyline length - explains why the TF-IDF space is sparse
    tokens = frame["cleaned_story"].fillna("").str.split().str.len()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(tokens, bins=40, color=PALETTE[0])
    ax.axvline(tokens.mean(), color="black", linestyle="--", linewidth=1,
               label=f"mean = {tokens.mean():.1f} tokens")
    ax.set_title("Storyline length after cleaning")
    ax.set_xlabel("Tokens per storyline")
    ax.set_ylabel("Movies")
    ax.legend()
    save(fig, "01_storyline_length.png")

    # 2. IMDb rating distribution
    ratings = pd.to_numeric(frame.get("imdb_rating"), errors="coerce").dropna()
    if len(ratings):
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.hist(ratings, bins=30, color=PALETTE[3])
        ax.set_title("IMDb rating distribution (2024 features)")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Movies")
        save(fig, "02_rating_distribution.png")

    # 3. Highest-weight TF-IDF terms across the corpus
    vectoriser = models.load_artifact(models.VECTORISER_PKL)
    matrix = models.load_artifact(models.MATRIX_PKL)
    weights = np.asarray(matrix.sum(axis=0)).ravel()
    vocab = np.array(vectoriser.get_feature_names_out())
    order = weights.argsort()[::-1][:20]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(vocab[order][::-1], weights[order][::-1], color=PALETTE[0])
    ax.set_title("Top 20 TF-IDF terms across the corpus")
    ax.set_xlabel("Summed TF-IDF weight")
    save(fig, "03_top_tfidf_terms.png")

    # 4. Similarity distribution - the honest picture of match strength
    similarity = models.load_artifact(models.SIMILARITY_PKL)
    values = np.asarray(similarity)
    upper = values[np.triu_indices_from(values, k=1)]
    neighbour_best = np.sort(values - np.eye(len(values)), axis=1)[:, -1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].hist(upper, bins=60, color=PALETTE[1])
    axes[0].set_title("All pairwise cosine similarities")
    axes[0].set_xlabel("Cosine similarity")
    axes[1].hist(neighbour_best, bins=40, color=PALETTE[2])
    axes[1].axvline(neighbour_best.mean(), color="black", linestyle="--",
                    linewidth=1, label=f"mean = {neighbour_best.mean():.3f}")
    axes[1].set_title("Best match per movie")
    axes[1].set_xlabel("Cosine similarity")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "04_similarity_distribution.png")

    # 5. Topic cluster sizes
    clustered = models.cluster_frame(frame)
    label_column = "topic" if "topic" in clustered.columns else "cluster"
    sizes = clustered[label_column].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh([str(v)[:52] for v in sizes.index], sizes.values,
            color=PALETTE[4])
    ax.set_title("Topic cluster sizes (browsing facets, not genres)")
    ax.set_xlabel("Movies")
    save(fig, "05_topic_clusters.png")

    print(f"\n{len(list(FIG_DIR.glob('*.png')))} figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
