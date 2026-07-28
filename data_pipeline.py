"""Data pipeline for the IMDb storyline recommendation system.

Responsibilities
----------------
1. Scrape movie names and storylines from IMDb with Selenium
   (``python data_pipeline.py --scrape``).
2. Fall back to a deterministic synthetic corpus with the identical schema
   when scraping is not run or the source is unreachable.
3. Clean the storyline text with an NLP pass: lower-casing, punctuation and
   digit stripping, stop-word removal and whitespace normalisation.
4. Persist the cleaned corpus to CSV and to MySQL through
   ``mysql-connector-python`` (cursor-based, no SQLAlchemy), with an
   automatic SQLite fallback.

Run standalone::

    python data_pipeline.py             # clean + load (uses cached/synthetic)
    python data_pipeline.py --scrape    # scrape IMDb first, then clean + load
"""

from __future__ import annotations

import argparse
import random
import re
import sqlite3
import sys
import time

import pandas as pd

import config

try:  # pragma: no cover - import guard only
    import mysql.connector
    from mysql.connector import Error as MySQLError

    MYSQL_AVAILABLE = True
except ImportError:  # pragma: no cover
    MYSQL_AVAILABLE = False

    class MySQLError(Exception):
        """Placeholder so except-clauses stay valid without the driver."""


# ---------------------------------------------------------------------------
# NLP helpers (self-contained: no NLTK download required at runtime)
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}

_PUNCTUATION = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case, strip punctuation and digits, and drop stop words."""
    if not isinstance(text, str):
        return ""
    lowered = text.lower()
    stripped = _PUNCTUATION.sub(" ", lowered)
    tokens = [
        token for token in _WHITESPACE.sub(" ", stripped).strip().split()
        if token not in STOP_WORDS and len(token) > 2
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------
class Database:
    """Cursor-based SQL wrapper over MySQL, falling back to SQLite."""

    def __init__(self) -> None:
        self.backend = "sqlite"
        self.conn = None
        self._connect()

    def _connect(self) -> None:
        backend = config.DB_BACKEND
        if backend in ("auto", "mysql") and MYSQL_AVAILABLE:
            try:
                self.conn = self._connect_mysql()
                self.backend = "mysql"
                print(f"[db] connected to MySQL {config.MYSQL_CONFIG['host']}:"
                      f"{config.MYSQL_CONFIG['port']}/"
                      f"{config.MYSQL_CONFIG['database']}")
                return
            except MySQLError as exc:
                if backend == "mysql":
                    raise
                print(f"[db] MySQL unavailable ({exc}); falling back to SQLite.")

        self.conn = sqlite3.connect(config.SQLITE_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.backend = "sqlite"
        print(f"[db] connected to SQLite at {config.SQLITE_PATH}")

    @staticmethod
    def _connect_mysql():
        cfg = dict(config.MYSQL_CONFIG)
        database = cfg.pop("database")
        bootstrap = mysql.connector.connect(connection_timeout=5, **cfg)
        cur = bootstrap.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        cur.close()
        bootstrap.close()
        return mysql.connector.connect(connection_timeout=5, database=database, **cfg)

    def _adapt(self, sql: str) -> str:
        return sql.replace("%s", "?") if self.backend == "sqlite" else sql

    def execute(self, sql: str, params: tuple = ()) -> None:
        cur = self.conn.cursor()
        cur.execute(self._adapt(sql), params)
        self.conn.commit()
        cur.close()

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        cur = self.conn.cursor()
        cur.executemany(self._adapt(sql), rows)
        self.conn.commit()
        cur.close()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        if self.backend == "mysql":
            cur = self.conn.cursor(dictionary=True)
            cur.execute(self._adapt(sql), params)
            rows = cur.fetchall()
        else:
            cur = self.conn.cursor()
            cur.execute(self._adapt(sql), params)
            rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()

    def create_schema(self) -> None:
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS movies (
                movie_id        INTEGER PRIMARY KEY,
                movie_name      VARCHAR(255) NOT NULL,
                storyline       TEXT,
                cleaned_story   TEXT,
                genre_hint      VARCHAR(60),
                release_year    INT
            )
            """
        )
        print("[db] schema ready (movies)")

    def load_movies(self, frame: pd.DataFrame) -> None:
        self.execute("DELETE FROM movies")
        rows = [
            (idx + 1, r["movie_name"], r["storyline"], r["cleaned_story"],
             r.get("genre_hint", ""), int(config.IMDB_YEAR))
            for idx, r in frame.reset_index(drop=True).iterrows()
        ]
        self.executemany(
            "INSERT INTO movies (movie_id, movie_name, storyline, "
            "cleaned_story, genre_hint, release_year) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )
        print(f"[db] loaded {len(rows)} movie rows")


# ---------------------------------------------------------------------------
# Selenium scraper
# ---------------------------------------------------------------------------
def _build_driver():
    """Configure Chrome so IMDb's WAF treats the session as a real browser."""
    import tempfile

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    if config.SCRAPE_HEADLESS:
        options.add_argument("--headless=new")
    for argument in (
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--window-size=1600,1000",
        "--disable-blink-features=AutomationControlled",
        f"--user-data-dir={tempfile.mkdtemp()}",
    ):
        options.add_argument(argument)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', "
                   "{get: () => undefined})"},
    )
    driver.set_page_load_timeout(config.SCRAPE_PAGE_LOAD_TIMEOUT)
    return driver


_CARD_SELECTOR = "li.ipc-metadata-list-summary-item"

# IMDb's markup shifts often. As of the current layout the title is no longer
# an <h3>; it is whatever element carries `.ipc-title__text`, and the metadata
# chips live inside `.dli-title-metadata` rather than `span.dli-title-metadata-item`.
# Everything is pulled in one JS pass: 1,000 cards x 5 findElement round-trips
# through the WebDriver protocol is slow and prone to stale-element errors,
# and `WebElement.text` returns "" for anything outside the viewport.
_EXTRACT_JS = """
return Array.from(
    document.querySelectorAll('li.ipc-metadata-list-summary-item')
).map(card => {
    const pick = sel => (card.querySelector(sel)?.textContent || '').trim();
    return {
        title: pick('.ipc-title__text'),
        storyline: pick('div.ipc-html-content-inner-div'),
        metadata: pick('.dli-title-metadata'),
        rating: pick('span.ipc-rating-star--rating'),
        votes: pick('span.ipc-rating-star--voteCount'),
        href: card.querySelector('a.ipc-title-link-wrapper')?.getAttribute('href') || ''
    };
});
"""

_YEAR_RE = re.compile(r"(19|20)\d{2}")
_RUNTIME_RE = re.compile(r"\d+h(?:\s*\d+m)?|\d+m")


def _expand_results(driver, target: int) -> int:
    """Click the "N more" button until enough cards are loaded."""
    from selenium.webdriver.common.by import By

    clicks = 0
    previous = len(driver.find_elements(By.CSS_SELECTOR, _CARD_SELECTOR))
    while previous < target and clicks < config.SCRAPE_MAX_CLICKS:
        try:
            button = driver.find_element(
                By.CSS_SELECTOR, "button.ipc-see-more__button"
            )
        except Exception:
            print("[scrape] no further 'see more' button; stopping expansion")
            break

        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center'});", button
        )
        time.sleep(0.4)
        try:
            driver.execute_script("arguments[0].click();", button)
        except Exception:
            break

        clicks += 1
        time.sleep(config.SCRAPE_CLICK_PAUSE)
        current = len(driver.find_elements(By.CSS_SELECTOR, _CARD_SELECTOR))
        print(f"[scrape] click {clicks}: {previous} -> {current} cards")
        if current == previous:  # Nothing new arrived; give up gracefully.
            break
        previous = current
    return previous


def _parse_card(raw: dict) -> dict | None:
    """Turn one raw JS payload into a clean record, or None if unusable."""
    title = re.sub(r"^\d+\.\s*", "", (raw.get("title") or "")).strip()
    storyline = (raw.get("storyline") or "").replace("\xa0", " ").strip()
    if not title or len(storyline.split()) < 5:
        return None

    metadata = (raw.get("metadata") or "").replace("\xa0", " ")
    year_match = _YEAR_RE.search(metadata)
    runtime_match = _RUNTIME_RE.search(metadata)
    votes = (raw.get("votes") or "").replace("\xa0", " ").strip(" ()")

    imdb_id = ""
    href = raw.get("href") or ""
    id_match = re.search(r"/title/(tt\d+)/", href)
    if id_match:
        imdb_id = id_match.group(1)

    return {
        "imdb_id": imdb_id,
        "movie_name": title,
        "storyline": storyline,
        "release_year": year_match.group(0) if year_match else "",
        "runtime": runtime_match.group(0) if runtime_match else "",
        "imdb_rating": (raw.get("rating") or "").strip(),
        "vote_count": votes,
    }


def scrape_imdb() -> pd.DataFrame:
    """Scrape movie names and storylines from IMDb's advanced-search page.

    Requires ``selenium`` plus a local Chrome install; Selenium Manager
    downloads a matching chromedriver automatically.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as ec
    from selenium.webdriver.support.ui import WebDriverWait

    driver = _build_driver()
    records: list[dict] = []
    try:
        print(f"[scrape] opening {config.IMDB_URL}")
        driver.get(config.IMDB_URL)
        time.sleep(config.SCRAPE_SETTLE_SECONDS)

        if "human verification" in (driver.title or "").lower():
            raise RuntimeError(
                "IMDb served its WAF human-verification challenge. Re-run "
                "with IMDB_HEADLESS=0 (the default) so a normal browser "
                "window is used."
            )

        WebDriverWait(driver, config.SCRAPE_TIMEOUT).until(
            ec.presence_of_element_located((By.CSS_SELECTOR, _CARD_SELECTOR))
        )
        print(f"[scrape] page title: {driver.title}")

        loaded = _expand_results(driver, config.SCRAPE_MAX_MOVIES)
        print(f"[scrape] {loaded} cards loaded; extracting")

        raw_rows = driver.execute_script(_EXTRACT_JS)
        for raw in raw_rows[: config.SCRAPE_MAX_MOVIES]:
            parsed = _parse_card(raw)
            if parsed:
                records.append(parsed)

        dropped = min(len(raw_rows), config.SCRAPE_MAX_MOVIES) - len(records)
        if dropped:
            print(f"[scrape] skipped {dropped} cards with no usable storyline")
    finally:
        driver.quit()

    frame = pd.DataFrame(records).drop_duplicates(subset=["movie_name"])
    print(f"[scrape] collected {len(frame)} movies with storylines "
          f"from IMDb {config.IMDB_YEAR}")
    if not frame.empty:
        frame.to_csv(config.RAW_CSV, index=False, encoding="utf-8")
        print(f"[scrape] saved -> {config.RAW_CSV.name}")
    return frame


# ---------------------------------------------------------------------------
# Synthetic corpus (fallback)
# ---------------------------------------------------------------------------
_TEMPLATES = {
    "Science Fiction": [
        "A {role} aboard a decaying colony ship discovers the {object} that "
        "the mission command has hidden for {number} years, forcing a choice "
        "between the crew and the {stake}.",
        "When an artificial intelligence begins rewriting its own {object}, a "
        "{role} on {place} must decide whether machine consciousness deserves "
        "the same protection as human life.",
        "Terraforming engineers on a distant moon uncover {object} beneath the "
        "ice, and the discovery threatens the fragile treaty holding the "
        "{stake} together.",
    ],
    "Thriller": [
        "A {role} with nothing left to lose tracks the {object} through {place} "
        "while the people who framed them close in from every direction.",
        "After a routine assignment goes wrong, a {role} realises the "
        "conspiracy runs through {place} and reaches the highest levels of "
        "the {stake}.",
        "A hostage negotiator has {number} hours to recover the {object} "
        "before a coordinated attack tears through {place}.",
    ],
    "Romance": [
        "Two strangers meet in {place} during the worst week of their lives "
        "and slowly build something neither of them believed they still "
        "wanted.",
        "A {role} returns home after {number} years and confronts the "
        "relationship they abandoned, along with the {stake} they left behind.",
        "Separated by circumstance and reunited by chance, a {role} and their "
        "first love rebuild a friendship in {place} while learning what "
        "forgiveness costs.",
    ],
    "Horror": [
        "A family renovating an old house in {place} unearths the {object}, "
        "and something that was buried with it begins to move through the "
        "walls at night.",
        "A {role} investigating disappearances in {place} finds that the "
        "victims all touched the same {object} in the {number} days before "
        "they vanished.",
        "Isolated by a storm, residents of {place} discover the {object} in "
        "the cellar and realise the thing hunting them wears familiar faces.",
    ],
    "Comedy": [
        "A hopelessly disorganised {role} accidentally inherits the {object} "
        "and must survive {number} chaotic days in {place} to keep it.",
        "Two rival {role}s are forced to share an apartment in {place}, and "
        "their escalating pranks spiral into an unlikely friendship.",
        "When a wedding in {place} collapses into disaster, a {role} improvises "
        "a rescue plan that makes everything spectacularly worse.",
    ],
    "Drama": [
        "A {role} confronting illness returns to {place} to repair the "
        "{stake} they spent {number} years avoiding.",
        "Across a single summer in {place}, a {role} learns that the {object} "
        "their family protected was never worth what it cost them.",
        "A quiet portrait of a {role} rebuilding a life in {place} after the "
        "collapse of everything that once defined them.",
    ],
    "Action": [
        "A retired {role} is pulled back into service when the {object} is "
        "stolen from a vault beneath {place}.",
        "Outgunned and out of time, a {role} fights across {place} to stop a "
        "mercenary force from selling the {object} to the highest bidder.",
        "A convoy carrying the {object} through {place} is ambushed, leaving "
        "one {role} to protect it for {number} brutal hours.",
    ],
    "Fantasy": [
        "A reluctant {role} inherits the {object} and must cross {place} to "
        "return it before the old bargain claims the {stake}.",
        "In a kingdom where magic is taxed, a {role} discovers the {object} "
        "can undo {number} generations of debt, and every faction wants it.",
        "A cartographer mapping the edges of {place} finds a door that opens "
        "onto the {object}, and the {stake} on both sides begins to bleed "
        "together.",
    ],
    "Mystery": [
        "A {role} reopens a cold case in {place} after the {object} surfaces "
        "in an evidence locker {number} years too late.",
        "When a guest vanishes from a sealed room in {place}, a {role} must "
        "untangle {number} contradictory testimonies to find the {object}.",
        "A journalist chasing the truth about the {object} in {place} learns "
        "that every witness has already been paid to forget.",
    ],
    "Historical": [
        "During a time of upheaval in {place}, a {role} risks everything to "
        "carry the {object} across the border and preserve the {stake}.",
        "Based on true events, a {role} spends {number} years documenting "
        "life in {place} while the world outside refuses to look.",
        "A forgotten {role} whose work shaped {place} is finally given the "
        "chronicle of the {stake} they built and lost.",
    ],
}

_ROLES = [
    "engineer", "detective", "archivist", "surgeon", "smuggler", "translator",
    "cartographer", "biologist", "journalist", "soldier", "teacher", "pilot",
    "curator", "diplomat", "musician", "botanist", "programmer", "chef",
]
_OBJECTS = [
    "encrypted ledger", "stolen prototype", "sealed manuscript", "black-box recorder",
    "family archive", "quarantine key", "forged treaty", "buried recording",
    "missing reactor core", "inherited map", "stolen vaccine", "cursed heirloom",
]
_PLACES = [
    "a flooded coastal city", "the northern mountains", "a decommissioned research station",
    "a border town", "a crumbling metropolis", "an orbital habitat",
    "a rain-soaked harbour", "a desert outpost", "an island monastery",
    "a shuttered mill town", "a sprawling arcology", "a frozen valley",
]
_STAKES = [
    "fragile alliance", "family legacy", "public trust", "last archive",
    "peace accord", "community they built", "future of the settlement",
    "truth they buried",
]
_TITLE_A = [
    "Silent", "Broken", "Crimson", "Distant", "Hollow", "Northern", "Last",
    "Quiet", "Burning", "Frozen", "Endless", "Bitter", "Golden", "Second",
    "Forgotten", "Restless", "Iron", "Pale",
]
_TITLE_B = [
    "Harbour", "Signal", "Archive", "Reckoning", "Meridian", "Covenant",
    "Threshold", "Cartography", "Inheritance", "Ascent", "Undertow",
    "Ledger", "Requiem", "Passage", "Aftermath", "Circuit", "Testament",
    "Horizon",
]


def generate_synthetic_corpus() -> pd.DataFrame:
    """Build a deterministic, clearly-fictional movie corpus.

    The schema matches the scraped output exactly (``movie_name``,
    ``storyline``), so no downstream code changes when the real scrape is run.
    Titles and plots are invented; no claims are made about real films.
    """
    rng = random.Random(config.RANDOM_SEED)
    seen: set[str] = set()
    records = []

    while len(records) < config.N_SYNTHETIC_MOVIES:
        genre = rng.choice(config.GENRES)
        title = f"{rng.choice(_TITLE_A)} {rng.choice(_TITLE_B)}"
        if rng.random() < 0.25:
            title += f" {rng.choice(['II', 'Part Two', 'Reborn', 'Rising'])}"
        if title in seen:
            continue
        seen.add(title)

        sentences = rng.sample(_TEMPLATES[genre], k=min(2, len(_TEMPLATES[genre])))
        storyline = " ".join(
            sentence.format(
                role=rng.choice(_ROLES),
                object=rng.choice(_OBJECTS),
                place=rng.choice(_PLACES),
                stake=rng.choice(_STAKES),
                number=rng.choice([three for three in (3, 5, 7, 12, 20, 30)]),
            )
            for sentence in sentences
        )
        records.append(
            {"movie_name": title, "storyline": storyline, "genre_hint": genre}
        )

    frame = pd.DataFrame(records)
    print(f"[data] generated {len(frame)} synthetic movie records")
    return frame


def load_raw_corpus() -> pd.DataFrame:
    if config.RAW_CSV.exists():
        print(f"[data] using cached corpus: {config.RAW_CSV.name}")
        return pd.read_csv(config.RAW_CSV)
    print("[data] no scraped corpus found (see DATASET_MISSING.txt); "
          "generating a synthetic corpus with the documented schema")
    frame = generate_synthetic_corpus()
    frame.to_csv(config.RAW_CSV, index=False)
    return frame


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_corpus(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop empties and duplicates, then NLP-clean every storyline."""
    frame = frame.copy()
    before = len(frame)

    frame["movie_name"] = frame["movie_name"].astype("string").str.strip()
    frame["storyline"] = frame["storyline"].astype("string").fillna("")
    if "genre_hint" not in frame.columns:
        frame["genre_hint"] = ""
    frame["genre_hint"] = frame["genre_hint"].astype("string").fillna("")

    frame = frame[frame["movie_name"].notna() & (frame["movie_name"] != "")]
    frame = frame.drop_duplicates(subset=["movie_name"], keep="first")
    frame = frame[frame["storyline"].str.split().str.len() >= 5]

    frame["cleaned_story"] = frame["storyline"].map(clean_text)
    frame = frame[frame["cleaned_story"].str.len() > 0].reset_index(drop=True)

    print(f"[clean] {before} -> {len(frame)} movies | "
          f"mean tokens per storyline = "
          f"{frame['cleaned_story'].str.split().str.len().mean():.1f}")
    return frame


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(scrape: bool = False) -> pd.DataFrame:
    raw = pd.DataFrame()
    if scrape:
        try:
            raw = scrape_imdb()
        except Exception as exc:
            print(f"[scrape] failed ({type(exc).__name__}: {exc}); "
                  "falling back to the cached/synthetic corpus")
    if raw.empty:
        raw = load_raw_corpus()

    cleaned = clean_corpus(raw)
    cleaned.to_csv(config.CLEANED_CSV, index=False)
    print(f"[data] cleaned corpus -> {config.CLEANED_CSV.name}")

    db = Database()
    try:
        db.create_schema()
        db.load_movies(cleaned)
        total = db.fetch_all("SELECT COUNT(*) AS n FROM movies")[0]["n"]
        print(f"[verify] movies row count = {total}")
    finally:
        db.close()
    return cleaned


def load_cleaned() -> pd.DataFrame:
    if config.CLEANED_CSV.exists():
        return pd.read_csv(config.CLEANED_CSV)
    return run_pipeline()


def main() -> int:
    parser = argparse.ArgumentParser(description="IMDb storyline data pipeline")
    parser.add_argument("--scrape", action="store_true",
                        help="scrape IMDb with Selenium before cleaning")
    args = parser.parse_args()

    print("=" * 70)
    print("IMDb Movie Recommendation System - data pipeline")
    print("=" * 70)
    run_pipeline(scrape=args.scrape)
    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
