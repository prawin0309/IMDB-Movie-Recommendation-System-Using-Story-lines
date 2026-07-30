"""Recommendation engine for the IMDb storyline project.

Engines
-------
* ``TfidfVectorizer`` over the cleaned storylines, persisted as a ``.pkl``.
* Full pairwise **cosine similarity** matrix, persisted as a ``.pkl``.
* ``KMeans`` topic clustering of the corpus for exploratory browsing.

Public API
----------
``recommend_by_text``   free-text storyline -> top-K similar movies
``recommend_by_title``  existing movie      -> top-K similar movies

Run standalone::

    python models.py
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import config
from data_pipeline import clean_text, load_cleaned

VECTORISER_PKL = config.ARTIFACT_DIR / "tfidf_vectorizer.pkl"
MATRIX_PKL = config.ARTIFACT_DIR / "tfidf_matrix.pkl"
SIMILARITY_PKL = config.ARTIFACT_DIR / "cosine_similarity.pkl"
KMEANS_PKL = config.ARTIFACT_DIR / "topic_kmeans.pkl"
TOPIC_METRICS_JSON = config.ARTIFACT_DIR / "topic_metrics.json"
TITLE_INDEX_PKL = config.ARTIFACT_DIR / "title_index.pkl"


def save_artifact(obj, path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)
    print(f"[save] {path.name}")


def load_artifact(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_vectoriser(frame: pd.DataFrame, method: str = "tfidf"):
    """Fit TF-IDF (default) or CountVectorizer over the cleaned storylines."""
    if method == "count":
        vectoriser = CountVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
        )
    else:
        vectoriser = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            sublinear_tf=True,
        )
    matrix = vectoriser.fit_transform(frame["cleaned_story"])
    return vectoriser, matrix


def train(frame: pd.DataFrame | None = None) -> dict:
    """Fit the vectoriser, similarity matrix and topic clusters."""
    frame = load_cleaned() if frame is None else frame

    vectoriser, matrix = build_vectoriser(frame)
    print(f"[tfidf] matrix shape = {matrix.shape} "
          f"({matrix.nnz} non-zero entries)")

    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)

    n_clusters = min(config.N_CLUSTERS, max(2, matrix.shape[0] // 20))
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED,
                    n_init=10)
    labels = kmeans.fit_predict(matrix)
    score = float(silhouette_score(matrix, labels)) if n_clusters > 1 else float("nan")

    title_index = {
        str(name).lower(): idx for idx, name in enumerate(frame["movie_name"])
    }

    save_artifact(vectoriser, VECTORISER_PKL)
    save_artifact(matrix, MATRIX_PKL)
    save_artifact(similarity, SIMILARITY_PKL)
    save_artifact(kmeans, KMEANS_PKL)
    save_artifact(title_index, TITLE_INDEX_PKL)

    # Persist the clustering quality so the UI can state it honestly instead of
    # presenting the topics as if they were well separated.
    TOPIC_METRICS_JSON.write_text(
        json.dumps({"n_clusters": n_clusters, "silhouette": score}, indent=2),
        encoding="utf-8",
    )

    print(f"[cluster] k={n_clusters}  silhouette={score:.4f}")
    print(f"[similarity] mean off-diagonal similarity = {similarity.mean():.4f}")
    if score < 0.05:
        print("[cluster] NOTE: IMDb plot blurbs average ~18 tokens, so the "
              "TF-IDF space is extremely sparse and KMeans finds little "
              "separation. The clusters are useful as browsing facets, not as "
              "hard genre boundaries. Recommendation quality is driven by "
              "pairwise cosine similarity, not by these clusters.")

    return {
        "vectoriser": vectoriser,
        "matrix": matrix,
        "similarity": similarity,
        "kmeans": kmeans,
        "labels": labels,
        "silhouette": score,
        "frame": frame,
    }


def artifacts_ready() -> bool:
    return all(
        path.exists()
        for path in (VECTORISER_PKL, MATRIX_PKL, SIMILARITY_PKL, TITLE_INDEX_PKL)
    )


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
def recommend_by_text(storyline: str, frame: pd.DataFrame | None = None,
                      top_k: int = config.TOP_K) -> pd.DataFrame:
    """Return the ``top_k`` movies whose storylines are closest to the input."""
    frame = load_cleaned() if frame is None else frame
    vectoriser = load_artifact(VECTORISER_PKL)
    matrix = load_artifact(MATRIX_PKL)

    cleaned = clean_text(storyline)
    if not cleaned:
        return pd.DataFrame(columns=["movie_name", "storyline", "similarity"])

    query = vectoriser.transform([cleaned])
    scores = cosine_similarity(query, matrix).ravel()
    order = np.argsort(scores)[::-1][:top_k]

    result = frame.iloc[order][["movie_name", "storyline"]].copy()
    result["similarity"] = scores[order].round(4)
    return result[result["similarity"] > 0].reset_index(drop=True)


def recommend_by_title(title: str, frame: pd.DataFrame | None = None,
                       top_k: int = config.TOP_K) -> pd.DataFrame:
    """Return the ``top_k`` movies most similar to an existing title."""
    frame = load_cleaned() if frame is None else frame
    title_index = load_artifact(TITLE_INDEX_PKL)
    similarity = load_artifact(SIMILARITY_PKL)

    position = title_index.get(str(title).lower())
    if position is None:
        return pd.DataFrame(columns=["movie_name", "storyline", "similarity"])

    scores = similarity[position]
    order = np.argsort(scores)[::-1][:top_k]
    result = frame.iloc[order][["movie_name", "storyline"]].copy()
    result["similarity"] = scores[order].round(4)
    return result.reset_index(drop=True)


def top_terms(top_n: int = 20) -> pd.DataFrame:
    """Highest-weight vocabulary terms across the whole corpus."""
    vectoriser = load_artifact(VECTORISER_PKL)
    matrix = load_artifact(MATRIX_PKL)
    weights = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectoriser.get_feature_names_out()
    return (
        pd.DataFrame({"term": terms, "weight": weights.round(3)})
        .sort_values("weight", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def cluster_frame(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the learned topic cluster (and its top terms) to each movie."""
    frame = load_cleaned() if frame is None else frame
    vectoriser = load_artifact(VECTORISER_PKL)
    matrix = load_artifact(MATRIX_PKL)
    kmeans = load_artifact(KMEANS_PKL)

    labelled = frame.copy()
    labelled["cluster"] = kmeans.predict(matrix)

    terms = np.array(vectoriser.get_feature_names_out())
    order = kmeans.cluster_centers_.argsort()[:, ::-1]
    names = {
        cid: ", ".join(terms[order[cid, :4]])
        for cid in range(kmeans.n_clusters)
    }
    labelled["topic"] = labelled["cluster"].map(names)
    return labelled


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("IMDb Movie Recommendation System - model training")
    print("=" * 70)
    frame = load_cleaned()
    print(f"[data] {len(frame)} movies loaded")

    result = train(frame)

    print("\nTop corpus terms")
    print(top_terms(15).to_string(index=False))

    demo_title = str(frame["movie_name"].iloc[0])
    print(f"\n[demo] movies similar to '{demo_title}'")
    print(recommend_by_title(demo_title, frame)[
        ["movie_name", "similarity"]].to_string(index=False))

    demo_text = (
        "A detective investigating a series of disappearances discovers a "
        "buried recording that links every victim to the same conspiracy."
    )
    print("\n[demo] recommendations for a free-text storyline")
    print(recommend_by_text(demo_text, frame)[
        ["movie_name", "similarity"]].to_string(index=False))

    print("\nTopic cluster sizes")
    print(cluster_frame(frame)["topic"].value_counts().to_string())

    print(f"\nSilhouette = {result['silhouette']:.4f}")
    print("\nModel training completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
