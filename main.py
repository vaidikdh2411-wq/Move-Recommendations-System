"""
Movie Recommendation System
Single-file ML project for the supplied MovieLens-style movies.csv.

Run:
    python main.py --run-all
    streamlit run main.py

The script:
1. Loads movies.csv
2. Performs data-quality checks
3. Saves a cleaned dataset
4. Generates before/after cleaning EDA charts
5. Creates an HTML EDA report
6. Builds a TF-IDF + cosine-similarity content recommender
7. Saves the trained artifacts
8. Provides a Streamlit recommendation UI

Important:
This CSV contains movieId, title, genres only. Therefore this is a
content-based recommender, not a personalized user-rating model.
"""

from __future__ import annotations

import argparse
import html
import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Matplotlib is imported only when reports/charts are generated.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "Dataset/movies.csv"
OUTPUT_DIR = BASE_DIR / "output"
CLEAN_DIR = OUTPUT_DIR / "cleaned_data"
BEFORE_DIR = OUTPUT_DIR / "before_cleaning"
AFTER_DIR = OUTPUT_DIR / "after_cleaning"
GRAPH_DIR = OUTPUT_DIR / "graphs"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

CLEAN_FILE = CLEAN_DIR / "cleaned_movies.csv"
MODEL_FILE = MODEL_DIR / "movie_recommender.pkl"
EDA_REPORT = REPORT_DIR / "EDA_Report.html"
QUALITY_REPORT = REPORT_DIR / "data_quality_report.html"


# ============================================================
# DIRECTORY SETUP
# ============================================================

def make_directories() -> None:
    for folder in [
        OUTPUT_DIR,
        CLEAN_DIR,
        BEFORE_DIR,
        AFTER_DIR,
        GRAPH_DIR,
        MODEL_DIR,
        REPORT_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_data(input_file: Path = DEFAULT_INPUT) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find {input_file}.\n"
            f"Place movies.csv in the same folder as main.py."
        )

    df = pd.read_csv(input_file)

    required = {"movieId", "title", "genres"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Expected: movieId, title, genres."
        )

    return df[["movieId", "title", "genres"]].copy()


# ============================================================
# DATA QUALITY
# ============================================================

def data_quality_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {
            str(k): int(v) for k, v in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_movie_ids": int(df["movieId"].duplicated().sum()),
        "unique_movie_ids": int(df["movieId"].nunique()),
        "unique_titles": int(df["title"].nunique()),
        "unique_genre_strings": int(df["genres"].nunique()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 2),
    }


def extract_year(title: str):
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else np.nan


# ============================================================
# CLEANING
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()

    clean["movieId"] = pd.to_numeric(clean["movieId"], errors="coerce")
    clean["title"] = clean["title"].astype("string").str.strip()
    clean["genres"] = clean["genres"].astype("string").str.strip()

    # Missing values
    clean = clean.dropna(subset=["movieId", "title", "genres"])

    # Remove invalid/empty values
    clean = clean[clean["title"].str.len() > 0]
    clean = clean[clean["genres"].str.len() > 0]

    # Remove placeholder genre if it is mixed with real genres.
    clean["genres"] = clean["genres"].str.replace(
        r"\(no genres listed\)", "Unknown", regex=True
    )

    # Normalize genre separator and whitespace.
    clean["genres"] = (
        clean["genres"]
        .str.replace("|", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Extract release year where available.
    clean["release_year"] = clean["title"].map(extract_year).astype("Int64")

    # Keep one record per movieId.
    clean = clean.drop_duplicates(subset=["movieId"], keep="first")

    # Keep one record per exact title+genre combination.
    clean = clean.drop_duplicates(subset=["title", "genres"], keep="first")

    clean["movieId"] = clean["movieId"].astype(int)

    clean = clean.sort_values("movieId").reset_index(drop=True)

    return clean


# ============================================================
# GENRE / YEAR HELPERS
# ============================================================

def genre_counts(df: pd.DataFrame) -> pd.Series:
    return (
        df["genres"]
        .fillna("Unknown")
        .str.split()
        .explode()
        .replace("", np.nan)
        .dropna()
        .value_counts()
    )


# ============================================================
# CHARTS
# ============================================================

def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str,
             path: Path, top_n: int = 15, horizontal: bool = True) -> None:
    s = series.head(top_n).sort_values()

    plt.figure(figsize=(10, 6))
    if horizontal:
        plt.barh(s.index.astype(str), s.values)
    else:
        plt.bar(s.index.astype(str), s.values)
        plt.xticks(rotation=45, ha="right")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def create_before_cleaning_charts(df: pd.DataFrame) -> list[Path]:
    paths = []

    missing = df.isna().sum().sort_values(ascending=False)
    p = BEFORE_DIR / "01_missing_values_before.png"
    save_bar(
        missing[missing > 0] if (missing > 0).any()
        else pd.Series({"No missing values": 0}),
        "Missing Values Before Cleaning",
        "Count",
        "Column",
        p,
        top_n=10,
    )
    paths.append(p)

    dup = pd.Series(
        {
            "Duplicate rows": int(df.duplicated().sum()),
            "Duplicate movie IDs": int(df["movieId"].duplicated().sum()),
        }
    )
    p = BEFORE_DIR / "02_duplicates_before.png"
    save_bar(dup.sort_values(), "Duplicate Records Before Cleaning",
             "Count", "Type", p, top_n=10)
    paths.append(p)

    genre = genre_counts(
        df.assign(genres=df["genres"].fillna("Unknown").str.replace("|", " ", regex=False))
    )
    p = BEFORE_DIR / "03_genres_before.png"
    save_bar(genre, "Top Genres Before Cleaning",
             "Number of Movies", "Genre", p)
    paths.append(p)

    return paths


def create_after_cleaning_charts(df: pd.DataFrame) -> list[Path]:
    paths = []

    genre = genre_counts(df)
    p = AFTER_DIR / "01_top_genres_after.png"
    save_bar(genre, "Top Genres After Cleaning",
             "Number of Movies", "Genre", p)
    paths.append(p)

    years = df["release_year"].dropna()
    if not years.empty:
        year_counts = years.value_counts().sort_index()
        plt.figure(figsize=(12, 6))
        plt.plot(year_counts.index, year_counts.values)
        plt.title("Movies by Release Year")
        plt.xlabel("Release Year")
        plt.ylabel("Number of Movies")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        p = AFTER_DIR / "02_movies_by_release_year.png"
        plt.savefig(p, dpi=160)
        plt.close()
        paths.append(p)

    genre_per_movie = df["genres"].str.split().str.len()
    plt.figure(figsize=(9, 5))
    plt.hist(genre_per_movie, bins=range(1, int(genre_per_movie.max()) + 2),
             edgecolor="black")
    plt.title("Number of Genres per Movie")
    plt.xlabel("Genres per Movie")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    p = AFTER_DIR / "03_genres_per_movie.png"
    plt.savefig(p, dpi=160)
    plt.close()
    paths.append(p)

    title_lengths = df["title"].str.len()
    plt.figure(figsize=(9, 5))
    plt.hist(title_lengths, bins=30, edgecolor="black")
    plt.title("Movie Title Length Distribution")
    plt.xlabel("Title Length")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    p = AFTER_DIR / "04_title_length_distribution.png"
    plt.savefig(p, dpi=160)
    plt.close()
    paths.append(p)

    return paths


def create_all_graphs(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    create_before_cleaning_charts(raw)
    create_after_cleaning_charts(clean)


# ============================================================
# TF-IDF CONTENT MODEL
# ============================================================

def build_model(clean: pd.DataFrame) -> dict:
    # Genre text is the strongest feature available in this CSV.
    # Adding title improves matching for movie-name/word similarity.
    feature_text = (
        clean["genres"].fillna("Unknown").astype(str) + " " +
        clean["title"]
        .fillna("")
        .astype(str)
        .str.replace(r"\(\d{4}\)", "", regex=True)
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[\w-]+\b",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(feature_text)

    # Sparse cosine similarity. For 62k movies, storing a full dense
    # 62k x 62k matrix would be unnecessarily huge. We calculate
    # similarity only for the selected movie at recommendation time.
    model = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "movie_ids": clean["movieId"].to_numpy(),
        "titles": clean["title"].tolist(),
        "genres": clean["genres"].tolist(),
    }

    return model


def save_model(model: dict) -> None:
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. Run: python main.py --run-all"
        )

    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def find_movie_index(clean: pd.DataFrame, query: str) -> int | None:
    q = str(query).strip().lower()

    exact = clean[clean["title"].str.lower() == q]
    if not exact.empty:
        return int(exact.index[0])

    contains = clean[
        clean["title"].str.lower().str.contains(re.escape(q), na=False)
    ]
    if not contains.empty:
        return int(contains.index[0])

    return None


def recommend_by_index(
    model: dict,
    movie_index: int,
    n: int = 10,
    exclude_same_title: bool = True,
) -> pd.DataFrame:
    matrix = model["tfidf_matrix"]

    # Calculate only one row of similarities instead of a full NxN matrix.
    scores = cosine_similarity(matrix[movie_index], matrix).ravel()

    # The selected movie itself should not be recommended.
    scores[movie_index] = -1

    order = np.argsort(-scores)

    rows = []
    selected_title = model["titles"][movie_index]

    for idx in order:
        if len(rows) >= n:
            break

        if exclude_same_title and model["titles"][idx] == selected_title:
            continue

        rows.append(
            {
                "movieId": int(model["movie_ids"][idx]),
                "title": model["titles"][idx],
                "genres": model["genres"][idx],
                "similarity_score": round(float(scores[idx]), 4),
            }
        )

    return pd.DataFrame(rows)


def recommend(
    model: dict,
    clean: pd.DataFrame,
    movie_title: str,
    n: int = 10,
) -> tuple[str | None, pd.DataFrame]:
    idx = find_movie_index(clean, movie_title)

    if idx is None:
        return None, pd.DataFrame()

    selected = clean.loc[idx, "title"]
    results = recommend_by_index(model, idx, n=n)

    return selected, results


# ============================================================
# AUTOMATED MODEL SANITY TEST
# ============================================================

def evaluate_recommender_sanity(
    model: dict,
    clean: pd.DataFrame,
    sample_movies: int = 20,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    This is not a supervised train/test accuracy score.
    The current dataset has no user ratings/interaction labels.

    We therefore perform a content-consistency sanity check:
    for sampled movies, calculate average similarity of their Top-K
    recommendations. This verifies that the recommendation engine
    is producing non-zero, content-related similarity scores.
    """
    if len(clean) == 0:
        return pd.DataFrame()

    indices = np.linspace(
        0, len(clean) - 1, min(sample_movies, len(clean)), dtype=int
    )

    records = []

    for idx in indices:
        recs = recommend_by_index(model, int(idx), n=top_k)
        if recs.empty:
            continue

        records.append(
            {
                "movie": clean.iloc[idx]["title"],
                "top_k": top_k,
                "mean_similarity": round(
                    float(recs["similarity_score"].mean()), 4
                ),
                "max_similarity": round(
                    float(recs["similarity_score"].max()), 4
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# REPORT GENERATION
# ============================================================

def table_html(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "<p>No data available.</p>"
    return df.head(max_rows).to_html(
        index=False,
        classes="data-table",
        border=0,
    )


def make_html_report(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    sanity: pd.DataFrame,
) -> None:
    raw_quality = data_quality_summary(raw)
    clean_quality = data_quality_summary(
        clean[["movieId", "title", "genres"]].copy()
    )

    genre = genre_counts(clean)
    year_available = int(clean["release_year"].notna().sum())

    before_files = sorted(BEFORE_DIR.glob("*.png"))
    after_files = sorted(AFTER_DIR.glob("*.png"))

    def image_cards(files):
        out = []
        for p in files:
            rel = p.relative_to(REPORT_DIR)
            # Report is in output/reports; images are one level above.
            src = "../" + str(p.relative_to(OUTPUT_DIR)).replace("\\", "/")
            out.append(
                f'<div class="chart"><h3>{html.escape(p.stem)}</h3>'
                f'<img src="{src}" alt="{html.escape(p.stem)}"></div>'
            )
        return "\n".join(out)

    html_doc = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Movie Recommendation System - EDA Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 0;
    background: #f5f7fa;
    color: #1f2937;
}}
.container {{
    width: 92%;
    max-width: 1250px;
    margin: 30px auto;
}}
.hero {{
    background: white;
    padding: 30px;
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
}}
h1, h2 {{ color: #111827; }}
.cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin: 20px 0;
}}
.card {{
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}}
.value {{
    font-size: 28px;
    font-weight: bold;
}}
.section {{
    background: white;
    padding: 24px;
    margin: 20px 0;
    border-radius: 14px;
}}
.chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 20px;
}}
.chart img {{
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 8px;
}}
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}}
.data-table th, .data-table td {{
    padding: 9px;
    border-bottom: 1px solid #ddd;
    text-align: left;
}}
.data-table th {{
    background: #f0f2f5;
}}
.note {{
    background: #fff8db;
    padding: 14px;
    border-left: 4px solid #d69e2e;
    border-radius: 5px;
}}
</style>
</head>
<body>
<div class="container">

<div class="hero">
<h1>Movie Recommendation System</h1>
<p><strong>EDA + Data Cleaning + Content-Based ML Recommendation</strong></p>
<p>
Dataset: MovieLens-style movie metadata. The supplied CSV contains
movie ID, title, and genres. It does not contain user ratings.
</p>
</div>

<div class="cards">
<div class="card"><div>Raw Movies</div><div class="value">{raw_quality["rows"]:,}</div></div>
<div class="card"><div>Clean Movies</div><div class="value">{clean_quality["rows"]:,}</div></div>
<div class="card"><div>Genres</div><div class="value">{genre.size}</div></div>
<div class="card"><div>Unique Titles</div><div class="value">{clean_quality["unique_titles"]:,}</div></div>
<div class="card"><div>Year Available</div><div class="value">{year_available:,}</div></div>
<div class="card"><div>Missing After Cleaning</div><div class="value">{clean_quality["missing_total"]}</div></div>
</div>

<div class="section">
<h2>1. Data Quality Before Cleaning</h2>
<ul>
<li>Rows: {raw_quality["rows"]:,}</li>
<li>Columns: {raw_quality["columns"]}</li>
<li>Missing values: {raw_quality["missing_total"]}</li>
<li>Duplicate rows: {raw_quality["duplicate_rows"]}</li>
<li>Duplicate movie IDs: {raw_quality["duplicate_movie_ids"]}</li>
<li>Memory usage: {raw_quality["memory_mb"]} MB</li>
</ul>
</div>

<div class="section">
<h2>2. Data Cleaning Performed</h2>
<ul>
<li>Validated required columns.</li>
<li>Converted movieId to numeric and removed invalid records.</li>
<li>Trimmed title and genre text.</li>
<li>Removed records with missing/empty movie information.</li>
<li>Normalized genre separators.</li>
<li>Extracted release year from movie titles when present.</li>
<li>Removed duplicate movie IDs.</li>
<li>Saved the cleaned dataset to <code>cleaned_data/cleaned_movies.csv</code>.</li>
</ul>
</div>

<div class="section">
<h2>3. After-Cleaning Quality</h2>
<ul>
<li>Clean rows: {clean_quality["rows"]:,}</li>
<li>Missing values: {clean_quality["missing_total"]}</li>
<li>Duplicate rows: {clean_quality["duplicate_rows"]}</li>
<li>Duplicate movie IDs: {clean_quality["duplicate_movie_ids"]}</li>
</ul>
</div>

<div class="section">
<h2>4. Before-Cleaning Charts</h2>
<div class="chart-grid">
{image_cards(before_files)}
</div>
</div>

<div class="section">
<h2>5. After-Cleaning EDA</h2>
<div class="chart-grid">
{image_cards(after_files)}
</div>
</div>

<div class="section">
<h2>6. Top Genres</h2>
{table_html(
    genre.rename("movie_count").reset_index().rename(columns={"index": "genre"}),
    20
)}
</div>

<div class="section">
<h2>7. Recommendation Model</h2>
<p>
The recommender uses <strong>TF-IDF</strong> feature extraction and
<strong>cosine similarity</strong>. Movie genre information and title text
are transformed into numerical feature vectors. Similar movies receive
higher cosine-similarity scores.
</p>
</div>

<div class="section">
<h2>8. Recommendation Sanity Check</h2>
<p>
Because the supplied dataset has no user/rating column, supervised
rating-prediction metrics such as RMSE cannot be legitimately calculated
from this file. The table below is a content-model sanity check, not
a claim of supervised prediction accuracy.
</p>
{table_html(sanity, 20)}
</div>

<div class="section">
<h2>9. Limitations</h2>
<div class="note">
For personalized recommendations, add a ratings file containing
<code>userId, movieId, rating, timestamp</code>. Then a collaborative or
hybrid recommender can be trained and evaluated using rating-prediction
and Top-N recommendation metrics.
</div>
</div>

<div class="section">
<h2>10. Output Files</h2>
<ul>
<li>Cleaned movie dataset</li>
<li>Before-cleaning charts</li>
<li>After-cleaning EDA charts</li>
<li>TF-IDF recommendation model</li>
<li>This HTML EDA report</li>
</ul>
</div>

</div>
</body>
</html>
"""

    EDA_REPORT.write_text(html_doc, encoding="utf-8")


def make_quality_report(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    before = data_quality_summary(raw)
    after = data_quality_summary(
        clean[["movieId", "title", "genres"]].copy()
    )

    report = {
        "dataset": "movies.csv",
        "before_cleaning": before,
        "after_cleaning": after,
        "cleaning_steps": [
            "Required-column validation",
            "Numeric movieId conversion",
            "Missing/empty row removal",
            "Whitespace normalization",
            "Genre separator normalization",
            "Release-year extraction",
            "Duplicate movieId removal",
            "Duplicate title+genre removal",
        ],
    }

    QUALITY_REPORT.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )


# ============================================================
# FULL PIPELINE
# ============================================================

def run_pipeline(input_file: Path = DEFAULT_INPUT) -> None:
    make_directories()

    print("\n" + "=" * 70)
    print("MOVIE RECOMMENDATION SYSTEM - FULL PIPELINE")
    print("=" * 70)

    print("\n[1/8] Loading dataset...")
    raw = load_data(input_file)
    print(f"Loaded: {len(raw):,} rows x {raw.shape[1]} columns")

    print("\n[2/8] Before-cleaning data quality...")
    before = data_quality_summary(raw)
    print(f"Missing values: {before['missing_total']:,}")
    print(f"Duplicate rows: {before['duplicate_rows']:,}")
    print(f"Duplicate movie IDs: {before['duplicate_movie_ids']:,}")

    print("\n[3/8] Creating before-cleaning charts...")
    create_before_cleaning_charts(raw)

    print("\n[4/8] Cleaning data...")
    clean = clean_data(raw)
    clean.to_csv(CLEAN_FILE, index=False)
    print(f"Clean dataset saved: {CLEAN_FILE}")
    print(f"Clean rows: {len(clean):,}")

    print("\n[5/8] Creating after-cleaning EDA...")
    create_after_cleaning_charts(clean)
    make_quality_report(raw, clean)

    print("\n[6/8] Building TF-IDF recommendation model...")
    model = build_model(clean)
    save_model(model)
    print(f"Model saved: {MODEL_FILE}")

    print("\n[7/8] Running recommendation sanity checks...")
    sanity = evaluate_recommender_sanity(model, clean)
    if not sanity.empty:
        print(
            f"Mean Top-K similarity: "
            f"{sanity['mean_similarity'].mean():.4f}"
        )

    print("\n[8/8] Creating HTML EDA report...")
    make_html_report(raw, clean, sanity)
    print(f"EDA report saved: {EDA_REPORT}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Clean data : {CLEAN_FILE}")
    print(f"Model      : {MODEL_FILE}")
    print(f"EDA report : {EDA_REPORT}")
    print(f"Quality    : {QUALITY_REPORT}")
    print(f"Graphs     : {GRAPH_DIR}")
    print("\nStart the web app with:")
    print("    streamlit run main.py")
    print("=" * 70 + "\n")


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
    )

    st.title("🎬 Movie Recommendation System")
    st.caption(
        "Content-based ML recommender using TF-IDF and cosine similarity"
    )

    # Auto-build if the user starts Streamlit before running the pipeline.
    if not CLEAN_FILE.exists() or not MODEL_FILE.exists():
        with st.spinner("Preparing data and recommendation model..."):
            run_pipeline(DEFAULT_INPUT)

    clean = pd.read_csv(CLEAN_FILE)
    model = load_model()

    st.sidebar.header("Recommendation Settings")
    n = st.sidebar.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
    )

    # Search box makes 62k movies easier to use.
    query = st.text_input(
        "Search for a movie",
        placeholder="Example: Toy Story",
    )

    if query:
        matches = clean[
            clean["title"].str.contains(
                re.escape(query), case=False, na=False
            )
        ].head(20)

        if matches.empty:
            st.warning("No matching movie found.")
            return

        options = matches["title"].tolist()
        selected = st.selectbox(
            "Select a movie",
            options,
        )

        if st.button("🎯 Recommend Similar Movies", type="primary"):
            selected_title, results = recommend(
                model,
                clean,
                selected,
                n=n,
            )

            if results.empty:
                st.error("Could not generate recommendations.")
                return

            st.subheader(f"Recommendations similar to: {selected_title}")

            for _, row in results.iterrows():
                score = float(row["similarity_score"])
                with st.container(border=True):
                    st.markdown(f"### 🎬 {row['title']}")
                    st.write(f"**Genres:** {row['genres']}")
                    st.progress(
                        min(max(score, 0.0), 1.0),
                        text=f"Similarity: {score:.2%}",
                    )

    st.divider()

    st.subheader("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", f"{len(clean):,}")
    c2.metric("Genres", f"{genre_counts(clean).size:,}")
    c3.metric("Unique Titles", f"{clean['title'].nunique():,}")
    c4.metric(
        "Years Available",
        f"{clean['release_year'].notna().sum():,}",
    )

    with st.expander("Top Genres"):
        top_genres = genre_counts(clean).head(15)
        st.bar_chart(top_genres)

    with st.expander("Sample Cleaned Data"):
        st.dataframe(clean.head(50), use_container_width=True)

    with st.expander("Project Files"):
        st.write(f"Clean dataset: `{CLEAN_FILE}`")
        st.write(f"Model: `{MODEL_FILE}`")
        st.write(f"EDA report: `{EDA_REPORT}`")
        st.write(f"Quality report: `{QUALITY_REPORT}`")

    st.info(
        "This version is content-based because the supplied CSV has no "
        "userId/rating interaction data. Add ratings.csv later for a "
        "personalized collaborative/hybrid model."
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Movie Recommendation System - single-file project"
    )

    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run cleaning, EDA, model training and report generation.",
    )

    parser.add_argument(
        "--recommend",
        type=str,
        default=None,
        help="Recommend movies similar to the supplied movie title.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of recommendations.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to movies.csv.",
    )

    args = parser.parse_args()

    input_file = Path(args.input)

    if args.run_all:
        run_pipeline(input_file)
        return

    if args.recommend:
        if not CLEAN_FILE.exists() or not MODEL_FILE.exists():
            run_pipeline(input_file)

        clean = pd.read_csv(CLEAN_FILE)
        model = load_model()

        selected, results = recommend(
            model,
            clean,
            args.recommend,
            n=args.n,
        )

        if selected is None:
            print(f"\nMovie not found: {args.recommend}")
            return

        print(f"\nRecommendations for: {selected}\n")
        print(results.to_string(index=False))
        return

    print(
        "\nMovie Recommendation System\n"
        "----------------------------\n"
        "1. Build everything:\n"
        "   python main.py --run-all\n\n"
        "2. Recommend from terminal:\n"
        '   python main.py --recommend "Toy Story (1995)" --n 10\n\n'
        "3. Start web application:\n"
        "   streamlit run main.py\n"
    )


if __name__ == "__main__":
    # Streamlit sets this environment variable when executing the app.
    try:
        import streamlit.runtime.scriptrunner as _st_runner
        IS_STREAMLIT = _st_runner.get_script_run_ctx() is not None
    except Exception:
        IS_STREAMLIT = False

    if IS_STREAMLIT:
        run_streamlit_app()
    else:
        main()
