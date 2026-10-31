# Movie Recommendation System
# Unit 1 + Unit 2 only
# Simple student-level EDA project

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# STEP 1 : DATA COLLECTION + DATA UNDERSTANDING
# ---------------------------------------------------------

# Keep movies.csv in the same folder as main.py
file = "Dataset/movies.csv"

df = pd.read_csv(file)

print("\n========== STEP 1 : DATA UNDERSTANDING ==========")

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nShape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nInformation:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique values:")
print(df.nunique())

# Numerical and categorical columns
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)

# ---------------------------------------------------------
# PROJECT UNDERSTANDING
# ---------------------------------------------------------

print("\n========== PROJECT UNDERSTANDING ==========")

print("Project Title: Movie Recommendation System")

print("\nProblem:")
print("Understand movie data and find useful patterns for a future")
print("movie recommendation system.")

print("\nObjective:")
print("1. Understand movie data.")
print("2. Clean the movie dataset.")
print("3. Perform EDA.")
print("4. Find genre and year patterns.")
print("5. Prepare clean data for future ML work.")

print("\nIndependent variables at EDA stage:")
print("movieId, title, genres, release_year, title_length, genre_count")

print("\nDependent variable:")
print("No dependent/target variable is defined at this Unit 1-2 stage.")
print("A future ML model will define the target according to its task.")

# ---------------------------------------------------------
# STEP 2 : BEFORE CLEANING REPORT
# ---------------------------------------------------------

print("\n========== STEP 2 : BEFORE CLEANING ==========")

os.makedirs("output", exist_ok=True)
os.makedirs("output/before_cleaning", exist_ok=True)
os.makedirs("output/after_cleaning", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)

before = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(df[c].dtype) for c in df.columns],
    "Missing": [df[c].isnull().sum() for c in df.columns],
    "Unique": [df[c].nunique() for c in df.columns]
})

before.to_csv(
    "output/before_cleaning_report.csv",
    index=False
)

print("\nBefore cleaning report:")
print(before)

# ---------------------------------------------------------
# STEP 3 : DATA CLEANING
# ---------------------------------------------------------

print("\n========== STEP 3 : DATA CLEANING ==========")

clean = df.copy()

# Convert movieId to number
clean["movieId"] = pd.to_numeric(
    clean["movieId"],
    errors="coerce"
)

# Remove missing required values
clean["title"] = clean["title"].astype("string").str.strip()
clean["genres"] = clean["genres"].astype("string").str.strip()

clean = clean.dropna(
    subset=["movieId", "title", "genres"]
)

# Remove empty text
clean = clean[
    (clean["title"] != "") &
    (clean["genres"] != "")
]

# Remove duplicate movie IDs
clean = clean.drop_duplicates(
    subset=["movieId"]
)

# Extract release year from title
def get_year(title):
    match = re.search(
        r"\((\d{4})\)\s*$",
        str(title)
    )
    if match:
        return int(match.group(1))
    return np.nan

clean["release_year"] = clean["title"].apply(get_year)

# Feature engineering for Unit 1 understanding
clean["title_length"] = clean["title"].str.len()

clean["title_word_count"] = (
    clean["title"]
    .str.replace(r"\(\d{4}\)", "", regex=True)
    .str.split()
    .str.len()
)

clean["genre_count"] = (
    clean["genres"]
    .str.split("|")
    .str.len()
)

# Save clean dataset
clean.to_csv(
    "output/cleaned_movies.csv",
    index=False
)

print("Original rows :", len(df))
print("Clean rows    :", len(clean))
print("Removed rows  :", len(df) - len(clean))

# ---------------------------------------------------------
# AFTER CLEANING REPORT
# ---------------------------------------------------------

after = pd.DataFrame({
    "Column": clean.columns,
    "Data_Type": [str(clean[c].dtype) for c in clean.columns],
    "Missing": [clean[c].isnull().sum() for c in clean.columns],
    "Unique": [clean[c].nunique() for c in clean.columns]
})

after.to_csv(
    "output/after_cleaning_report.csv",
    index=False
)

# Before vs after
comparison = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Missing Values",
        "Duplicate Rows",
        "Duplicate Movie IDs"
    ],
    "Before": [
        len(df),
        df.shape[1],
        df.isnull().sum().sum(),
        df.duplicated().sum(),
        df["movieId"].duplicated().sum()
    ],
    "After": [
        len(clean),
        clean.shape[1],
        clean.isnull().sum().sum(),
        clean.duplicated().sum(),
        clean["movieId"].duplicated().sum()
    ]
})

comparison.to_csv(
    "output/before_after_comparison.csv",
    index=False
)

print("\nBefore vs After:")
print(comparison)

# ---------------------------------------------------------
# STEP 4 : UNIT 1 EDA - CENTRAL TENDENCY
# ---------------------------------------------------------

print("\n========== STEP 4 : UNIT 1 EDA ==========")

numeric = clean[
    [
        "movieId",
        "release_year",
        "title_length",
        "title_word_count",
        "genre_count"
    ]
]

print("\nMean:")
print(numeric.mean(numeric_only=True))

print("\nMedian:")
print(numeric.median(numeric_only=True))

print("\nMode:")
print(numeric.mode(numeric_only=True).iloc[0])

print("\nVariance:")
print(numeric.var(numeric_only=True))

print("\nStandard Deviation:")
print(numeric.std(numeric_only=True))

print("\nRange:")
print(
    numeric.max(numeric_only=True)
    - numeric.min(numeric_only=True)
)

print("\nSkewness:")
print(numeric.skew(numeric_only=True))

print("\nKurtosis:")
print(numeric.kurt(numeric_only=True))

# Save statistics
stats = pd.DataFrame({
    "Mean": numeric.mean(),
    "Median": numeric.median(),
    "Variance": numeric.var(),
    "Standard_Deviation": numeric.std(),
    "Skewness": numeric.skew(),
    "Kurtosis": numeric.kurt()
})

stats.to_csv(
    "output/numerical_statistics.csv"
)

# ---------------------------------------------------------
# UNIT 1 : NUMERICAL / CATEGORICAL SUMMARY
# ---------------------------------------------------------

data_type_report = pd.DataFrame({
    "Column": clean.columns,
    "Type": [
        "Numerical" if pd.api.types.is_numeric_dtype(clean[c])
        else "Categorical/Text"
        for c in clean.columns
    ]
})

data_type_report.to_csv(
    "output/data_type_report.csv",
    index=False
)

# ---------------------------------------------------------
# UNIT 1 : OUTLIER CHECK USING IQR
# ---------------------------------------------------------

outlier_result = []

for col in [
    "release_year",
    "title_length",
    "title_word_count",
    "genre_count"
]:
    values = clean[col].dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = ((values < lower) | (values > upper)).sum()

    outlier_result.append({
        "Column": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Limit": lower,
        "Upper_Limit": upper,
        "Outlier_Count": count
    })

outliers = pd.DataFrame(outlier_result)

print("\nOutlier report:")
print(outliers)

outliers.to_csv(
    "output/outlier_report.csv",
    index=False
)

# ---------------------------------------------------------
# STEP 5 : EDA GRAPHS
# ---------------------------------------------------------

print("\n========== STEP 5 : EDA GRAPHS ==========")

# 1. Histogram
plt.figure(figsize=(8, 5))
plt.hist(
    clean["title_length"],
    bins=30,
    edgecolor="black"
)
plt.title("Histogram - Movie Title Length")
plt.xlabel("Title Length")
plt.ylabel("Number of Movies")
plt.savefig(
    "output/after_cleaning/01_histogram.png"
)
plt.close()

# 2. Box Plot
plt.figure(figsize=(8, 5))
plt.boxplot(
    clean["title_length"].dropna()
)
plt.title("Box Plot - Movie Title Length")
plt.ylabel("Title Length")
plt.savefig(
    "output/after_cleaning/02_boxplot.png"
)
plt.close()

# 3. Genre Bar Chart
genre_count = (
    clean["genres"]
    .str.split("|")
    .explode()
    .value_counts()
    .head(15)
)

plt.figure(figsize=(10, 6))
plt.bar(
    genre_count.index,
    genre_count.values
)
plt.title("Top 15 Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    "output/after_cleaning/03_genre_bar_chart.png"
)
plt.close()

# 4. Pie Chart
pie_data = genre_count.head(8)

plt.figure(figsize=(8, 8))
plt.pie(
    pie_data.values,
    labels=pie_data.index,
    autopct="%1.1f%%"
)
plt.title("Top Movie Genre Distribution")
plt.savefig(
    "output/after_cleaning/04_genre_pie_chart.png"
)
plt.close()

# 5. Movies by year
year_count = (
    clean["release_year"]
    .dropna()
    .astype(int)
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 5))
plt.plot(
    year_count.index,
    year_count.values
)
plt.title("Movies by Release Year")
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.grid()
plt.tight_layout()
plt.savefig(
    "output/after_cleaning/05_movies_by_year.png"
)
plt.close()

# 6. Scatter Plot
sample = clean.dropna(
    subset=["release_year", "title_length"]
).sample(
    min(5000, len(clean)),
    random_state=42
)

plt.figure(figsize=(8, 5))
plt.scatter(
    sample["release_year"],
    sample["title_length"],
    alpha=0.4
)
plt.title(
    "Bivariate Analysis: Release Year vs Title Length"
)
plt.xlabel("Release Year")
plt.ylabel("Title Length")
plt.savefig(
    "output/after_cleaning/06_scatter_plot.png"
)
plt.close()

# 7. Correlation Matrix
corr = numeric.corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(
    "output/after_cleaning/07_correlation_matrix.png"
)
plt.close()

print("Histogram: completed")
print("Box Plot: completed")
print("Bar Chart: completed")
print("Pie Chart: completed")
print("Line Chart: completed")
print("Scatter Plot: completed")
print("Correlation Matrix: completed")

# ---------------------------------------------------------
# BEFORE-CLEANING BASIC GRAPH
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
missing = df.isnull().sum()

plt.bar(
    missing.index,
    missing.values
)

plt.title("Missing Values Before Cleaning")
plt.xlabel("Column")
plt.ylabel("Missing Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(
    "output/before_cleaning/01_missing_values.png"
)
plt.close()

# ---------------------------------------------------------
# STEP 6 : AUTOMATED EDA - UNIT 2
# ---------------------------------------------------------

print("\n========== STEP 6 : AUTOMATED EDA ==========")

# Unit 2 specifically teaches automated EDA.
# Current material recommends ydata_profiling instead of the
# older pandas-profiling package.

try:
    from ydata_profiling import ProfileReport

    profile = ProfileReport(
        clean,
        title="Movie Recommendation System - Automated EDA",
        explorative=True
    )

    profile.to_file(
        "output/reports/Automated_EDA_Report.html"
    )

    print(
        "Automated EDA created using ydata-profiling."
    )

except Exception as e:
    print(
        "\nydata-profiling is not installed."
    )
    print(
        "Install it with:"
    )
    print(
        "pip install ydata-profiling"
    )

    # Simple fallback report
    with open(
        "output/reports/Automated_EDA_Report.html",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("""
        <html>
        <head>
        <title>Movie Automated EDA</title>
        </head>
        <body>
        <h1>Movie Recommendation System - Automated EDA</h1>
        <p>Install ydata-profiling to generate the complete automated report.</p>
        </body>
        </html>
        """)

# ---------------------------------------------------------
# STEP 7 : FINAL EDA REPORT
# ---------------------------------------------------------

print("\n========== STEP 7 : FINAL EDA REPORT ==========")

report = """
<html>
<head>
<title>Movie Recommendation System - EDA Report</title>

<style>
body {
    font-family: Arial;
    margin: 30px;
}

table {
    border-collapse: collapse;
    width: 100%;
}

th, td {
    border: 1px solid #999;
    padding: 8px;
}

img {
    max-width: 800px;
    margin: 15px;
}
</style>

</head>

<body>

<h1>Movie Recommendation System</h1>

<h2>1. Dataset Understanding</h2>
<p>
The dataset contains movie information with movie ID, movie title
and movie genres.
</p>

<h2>2. Data Cleaning</h2>
<ul>
<li>Checked missing values.</li>
<li>Checked duplicate rows.</li>
<li>Checked duplicate movie IDs.</li>
<li>Cleaned text columns.</li>
<li>Extracted release year.</li>
<li>Created title length.</li>
<li>Created title word count.</li>
<li>Created genre count.</li>
</ul>

<h2>3. Numerical Data</h2>
<p>
Numerical variables used for EDA include movieId, release_year,
title_length, title_word_count and genre_count.
</p>

<h2>4. Categorical/Text Data</h2>
<p>
Title and genres are text/categorical information.
Genres can contain multiple categories for one movie.
</p>

<h2>5. Independent and Dependent Variables</h2>
<p>
At this Unit 1 and Unit 2 EDA stage, there is no machine-learning
dependent/target variable. The dataset is being studied and prepared
before model building.
</p>

<p>
Possible analysis variables are movieId, release_year, title_length,
title_word_count and genre_count.
</p>

<h2>6. Univariate Analysis</h2>
<p>
Histogram, box plot and genre count analysis were performed.
</p>

<h2>7. Bivariate Analysis</h2>
<p>
Release year and title length were compared using a scatter plot.
</p>

<h2>8. Multivariate Analysis</h2>
<p>
A correlation matrix was created for numerical variables.
</p>

<h2>9. Central Tendency and Statistics</h2>
<p>
Mean, median, mode, variance, standard deviation, range, skewness
and kurtosis were calculated.
</p>

<h2>10. Output Graphs</h2>

<img src="../after_cleaning/01_histogram.png">
<img src="../after_cleaning/02_boxplot.png">
<img src="../after_cleaning/03_genre_bar_chart.png">
<img src="../after_cleaning/04_genre_pie_chart.png">
<img src="../after_cleaning/05_movies_by_year.png">
<img src="../after_cleaning/06_scatter_plot.png">
<img src="../after_cleaning/07_correlation_matrix.png">

<h2>11. Automated EDA</h2>
<p>
A separate Automated_EDA_Report.html file is generated using
ydata-profiling.
</p>

</body>
</html>
"""

with open(
    "output/reports/EDA_Report.html",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

# ---------------------------------------------------------
# FINAL MESSAGE
# ---------------------------------------------------------

print("\n==========================================")
print("UNIT 1 + UNIT 2 PROJECT COMPLETED")
print("==========================================")

print("\nCreated files:")

print("output/cleaned_movies.csv")
print("output/before_cleaning_report.csv")
print("output/after_cleaning_report.csv")
print("output/before_after_comparison.csv")
print("output/data_type_report.csv")
print("output/numerical_statistics.csv")
print("output/outlier_report.csv")

print("\nGraphs:")
print("output/before_cleaning/")
print("output/after_cleaning/")

print("\nReports:")
print("output/reports/EDA_Report.html")
print("output/reports/Automated_EDA_Report.html")

print("\nUnit 1 and Unit 2 are completed.")
print("Unit 3 ML model is NOT performed.")
