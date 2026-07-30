"""Streamlit application for the IMDb storyline recommendation system.

Pages
-----
Recommend by Storyline · Recommend by Movie · Browse Corpus ·
Topic Clusters · Corpus Insights

Run::

    streamlit run app.py
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

import config
import models
from data_pipeline import clean_text, load_cleaned

st.set_page_config(
    page_title="IMDb Storyline Recommender",
    page_icon="🎬",
    layout="wide",
)

PAGES = [
    "Recommend by Storyline",
    "Recommend by Movie",
    "Browse Corpus",
    "Topic Clusters",
    "Corpus Insights",
]

EXAMPLE_STORYLINE = (
    "A detective investigating a series of disappearances in a small town "
    "uncovers a buried recording that links every victim to the same "
    "conspiracy, and the people who paid the witnesses to forget."
)


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading movie corpus…")
def get_corpus() -> pd.DataFrame:
    return load_cleaned()


def artefacts_warning() -> None:
    st.warning(
        "Model artefacts not found. Run `python data_pipeline.py` then "
        "`python models.py` before using this page."
    )


def render_recommendations(result: pd.DataFrame) -> None:
    """Render a recommendation table as readable cards plus a bar chart."""
    if result.empty:
        st.info("No sufficiently similar movies found. Try a longer storyline.")
        return

    for rank, row in enumerate(result.itertuples(index=False), start=1):
        with st.container(border=True):
            head, score = st.columns([5, 1])
            head.markdown(f"### {rank}. {row.movie_name}")
            facts = []
            for attribute, label in (("release_year", ""),
                                     ("runtime", ""),
                                     ("imdb_rating", "★"),
                                     ("vote_count", "votes")):
                value = getattr(row, attribute, "")
                if isinstance(value, str) and value.strip():
                    facts.append(f"{label} {value}".strip())
            if facts:
                head.caption(" · ".join(facts))
            score.metric("Similarity", f"{row.similarity:.3f}")
            st.write(row.storyline)

    st.plotly_chart(
        px.bar(result.sort_values("similarity"), x="similarity", y="movie_name",
               orientation="h", color="similarity",
               color_continuous_scale="Viridis",
               title="Cosine similarity of recommended movies"),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_by_storyline(corpus: pd.DataFrame) -> None:
    st.header("🎬 Recommend by Storyline")
    st.caption(
        "Enter a plot description. The text is NLP-cleaned, vectorised with "
        "TF-IDF and compared against every movie using cosine similarity."
    )

    text = st.text_area("Storyline", value=EXAMPLE_STORYLINE, height=150)
    top_k = st.slider("Number of recommendations", 3, 15, config.TOP_K)

    if st.button("Recommend", type="primary", use_container_width=True):
        if not models.artifacts_ready():
            artefacts_warning()
            return
        cleaned = clean_text(text)
        if not cleaned:
            st.error("The storyline contains no usable keywords.")
            return
        with st.expander("Cleaned query text (stop words removed)"):
            st.code(cleaned)
        render_recommendations(models.recommend_by_text(text, corpus, top_k))


def page_by_movie(corpus: pd.DataFrame) -> None:
    st.header("🍿 Recommend by Movie")
    st.caption("Pick a movie already in the corpus to find its closest matches.")

    title = st.selectbox("Movie", sorted(corpus["movie_name"].astype(str)))
    top_k = st.slider("Number of recommendations", 3, 15, config.TOP_K,
                      key="title_k")

    selected = corpus[corpus["movie_name"] == title].iloc[0]
    with st.container(border=True):
        st.markdown(f"**Selected storyline — {title}**")
        st.write(selected["storyline"])

    if st.button("Find similar movies", type="primary", use_container_width=True):
        if not models.artifacts_ready():
            artefacts_warning()
            return
        render_recommendations(models.recommend_by_title(title, corpus, top_k))


def page_browse(corpus: pd.DataFrame) -> None:
    st.header("📚 Browse Corpus")
    search = st.text_input("Search titles and storylines")
    view = corpus
    if search:
        mask = (
            corpus["movie_name"].astype(str).str.contains(search, case=False, na=False)
            | corpus["storyline"].astype(str).str.contains(search, case=False, na=False)
        )
        view = corpus[mask]

    st.caption(f"{len(view)} of {len(corpus)} movies")
    display_columns = [
        column for column in
        ["movie_name", "release_year", "imdb_rating", "vote_count",
         "runtime", "storyline"]
        if column in view.columns
    ]
    st.dataframe(
        view[display_columns],
        use_container_width=True, hide_index=True, height=520,
    )


def page_clusters(corpus: pd.DataFrame) -> None:
    st.header("🧩 Topic Clusters")
    if not models.KMEANS_PKL.exists():
        artefacts_warning()
        return

    labelled = models.cluster_frame(corpus)
    sizes = labelled["topic"].value_counts().reset_index()
    sizes.columns = ["topic", "movies"]

    # Be explicit about how weak this structure is. TF-IDF vectors over short
    # storylines are extremely sparse, so KMeans finds only marginal
    # separation. The groups below are a browsing aid, not a claim that the
    # corpus contains eight distinct genres.
    if models.TOPIC_METRICS_JSON.exists():
        topic_metrics = json.loads(
            models.TOPIC_METRICS_JSON.read_text(encoding="utf-8")
        )
        st.warning(
            f"Silhouette score **{topic_metrics['silhouette']:.4f}** across "
            f"{topic_metrics['n_clusters']} clusters. A value this close to "
            "zero means the topics are only marginally separated - TF-IDF "
            "vectors over short storylines are too sparse for clean "
            "clustering. Treat these groups as a browsing aid, not as "
            "well-defined genres. The recommender itself does not use them."
        )

    st.plotly_chart(
        px.bar(sizes, x="movies", y="topic", orientation="h", color="movies",
               color_continuous_scale="Teal",
               title="Movies per discovered topic (top TF-IDF terms shown)"),
        use_container_width=True,
    )

    chosen = st.selectbox("Inspect a topic", sizes["topic"])
    st.dataframe(
        labelled[labelled["topic"] == chosen][["movie_name", "storyline"]],
        use_container_width=True, hide_index=True, height=420,
    )


def page_insights(corpus: pd.DataFrame) -> None:
    st.header("📈 Corpus Insights")

    cols = st.columns(4)
    lengths = corpus["storyline"].astype(str).str.split().str.len()
    cols[0].metric("Movies", f"{len(corpus):,}")
    cols[1].metric("Mean storyline length", f"{lengths.mean():.0f} words")
    cols[2].metric("Unique cleaned tokens",
                   f"{corpus['cleaned_story'].astype(str).str.split().explode().nunique():,}")
    if "imdb_rating" in corpus.columns:
        ratings = pd.to_numeric(corpus["imdb_rating"], errors="coerce")
        cols[3].metric("Mean IMDb rating", f"{ratings.mean():.2f}")

    if "imdb_rating" in corpus.columns:
        rated = corpus.assign(
            rating=pd.to_numeric(corpus["imdb_rating"], errors="coerce")
        ).dropna(subset=["rating"])
        if not rated.empty:
            st.plotly_chart(
                px.histogram(rated, x="rating", nbins=30,
                             title="IMDb rating distribution across the corpus"),
                use_container_width=True,
            )

    st.plotly_chart(
        px.histogram(lengths.to_frame("words"), x="words", nbins=30,
                     title="Storyline length distribution"),
        use_container_width=True,
    )

    if models.artifacts_ready():
        terms = models.top_terms(25)
        st.plotly_chart(
            px.bar(terms.sort_values("weight"), x="weight", y="term",
                   orientation="h",
                   title="Highest-weight TF-IDF terms across the corpus"),
            use_container_width=True,
        )
    else:
        artefacts_warning()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main() -> None:
    st.sidebar.title("🎬 Storyline Recommender")
    choice = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.divider()
    st.sidebar.caption(
        f"TF-IDF + cosine similarity over {len(get_corpus()):,} IMDb "
        f"{config.IMDB_YEAR} storylines. Run "
        "`python data_pipeline.py --scrape` to refresh from IMDb."
    )

    corpus = get_corpus()

    if choice == "Recommend by Storyline":
        page_by_storyline(corpus)
    elif choice == "Recommend by Movie":
        page_by_movie(corpus)
    elif choice == "Browse Corpus":
        page_browse(corpus)
    elif choice == "Topic Clusters":
        page_clusters(corpus)
    elif choice == "Corpus Insights":
        page_insights(corpus)


if __name__ == "__main__":
    main()
