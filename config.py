"""Configuration for the IMDb storyline recommendation system."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

for _folder in (DATA_DIR, ARTIFACT_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "imdb_2024_movies.csv"
CLEANED_CSV = DATA_DIR / "cleaned_movies.csv"
SQLITE_PATH = DATA_DIR / "guvi_db.sqlite3"

MYSQL_CONFIG = {
    "host": os.getenv("IMDB_DB_HOST", "localhost"),
    "port": int(os.getenv("IMDB_DB_PORT", "3306")),
    "user": os.getenv("IMDB_DB_USER", "root"),
    "password": os.getenv("IMDB_DB_PASSWORD", "root"),
    "database": os.getenv("IMDB_DB_NAME", "guvi_db"),
}
DB_BACKEND = os.getenv("IMDB_DB_BACKEND", "auto").lower()

# --- Scraping ---------------------------------------------------------------
IMDB_YEAR = 2024
IMDB_URL = (
    "https://www.imdb.com/search/title/"
    f"?title_type=feature&release_date={IMDB_YEAR}-01-01,{IMDB_YEAR}-12-31"
)
SCRAPE_MAX_MOVIES = 1000

# IMDb sits behind an AWS WAF that serves a "Human Verification" challenge to
# headless Chrome. A normal visible browser window passes it, so the scraper
# runs non-headless by default. Set IMDB_HEADLESS=1 to override.
SCRAPE_HEADLESS = os.getenv("IMDB_HEADLESS", "0") == "1"
SCRAPE_TIMEOUT = 30
SCRAPE_PAGE_LOAD_TIMEOUT = 90
SCRAPE_SETTLE_SECONDS = 6
SCRAPE_CLICK_PAUSE = 2.5
SCRAPE_MAX_CLICKS = 40

# --- Modelling --------------------------------------------------------------
RANDOM_SEED = 42
N_SYNTHETIC_MOVIES = 400
TOP_K = 5
TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM_RANGE = (1, 2)
# Scraped IMDb plot blurbs average ~18 tokens. min_df=2 drops hapax terms,
# which can never contribute to a movie-to-movie match, and max_df=0.4 drops
# terms shared by more than 40% of blurbs so common plot filler does not
# dominate the cosine scores.
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.4
N_CLUSTERS = 8

GENRES = [
    "Science Fiction", "Thriller", "Romance", "Horror", "Comedy",
    "Drama", "Action", "Fantasy", "Mystery", "Historical",
]
