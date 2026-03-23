from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "spotify_tracks.csv"
OUTPUT_DIR = BASE_DIR / "visualizations"


def load_data() -> pd.DataFrame:
    """Load the Spotify dataset and coerce numeric columns."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    numeric_columns = [
        "release_year",
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "key",
        "mode",
        "time_signature",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    text_columns = ["track_name", "artist_name", "genre"]
    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    return df.dropna(subset=["genre", "popularity", "release_year", "tempo"])


def build_summaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recreate the outputs from Lab 2 using pandas."""
    genre_summary = (
        df[df["popularity"] > 50]
        .groupby("genre", as_index=False)
        .agg(
            total_tracks=("track_id", "count"),
            avg_popularity=("popularity", "mean"),
        )
        .sort_values("avg_popularity", ascending=False)
    )

    year_summary = (
        df[df["release_year"] >= 2015]
        .groupby("release_year", as_index=False)
        .agg(
            total_tracks=("track_id", "count"),
            avg_tempo=("tempo", "mean"),
        )
        .sort_values("release_year")
    )

    rock_songs = (
        df[df["genre"].str.lower() == "rock"][
            ["track_name", "artist_name", "genre", "popularity"]
        ]
        .sort_values("popularity", ascending=False)
        .head(10)
    )

    return genre_summary, year_summary, rock_songs


def save_plot(filename: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def create_matplotlib_visuals(
    df: pd.DataFrame,
    genre_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    rock_songs: pd.DataFrame,
) -> None:
    """Create 5 visualizations using matplotlib."""
    plt.style.use("ggplot")
    top_genres = genre_summary.head(10)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        top_genres["genre"],
        top_genres["total_tracks"],
        color=plt.cm.Set3(range(len(top_genres))),
        edgecolor="black",
    )
    plt.title("Matplotlib 1: Tracks Above Popularity 50 by Genre", fontsize=14, weight="bold")
    plt.xlabel("Genre")
    plt.ylabel("Number of Tracks")
    plt.xticks(rotation=45, ha="right")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 5,
            f"{int(height)}",
            ha="center",
            fontsize=9,
        )
    save_plot("matplotlib_1_genre_total_tracks.png")

    plt.figure(figsize=(12, 6))
    plt.plot(
        year_summary["release_year"],
        year_summary["avg_tempo"],
        marker="o",
        linewidth=2.5,
        color="#2a9d8f",
    )
    plt.title("Matplotlib 2: Average Tempo by Release Year (2015 Onward)", fontsize=14, weight="bold")
    plt.xlabel("Release Year")
    plt.ylabel("Average Tempo")
    plt.grid(alpha=0.3, linestyle="--")
    save_plot("matplotlib_2_avg_tempo_by_year.png")

    plt.figure(figsize=(12, 6))
    plt.barh(
        rock_songs["track_name"],
        rock_songs["popularity"],
        color="#e76f51",
        edgecolor="black",
    )
    plt.title("Matplotlib 3: Top 10 Rock Songs by Popularity", fontsize=14, weight="bold")
    plt.xlabel("Popularity")
    plt.ylabel("Track Name")
    plt.gca().invert_yaxis()
    save_plot("matplotlib_3_top_rock_songs.png")

    plt.figure(figsize=(12, 6))
    plt.hist(
        df["popularity"].dropna(),
        bins=20,
        color="#457b9d",
        edgecolor="white",
        alpha=0.9,
    )
    plt.title("Matplotlib 4: Popularity Distribution of Spotify Tracks", fontsize=14, weight="bold")
    plt.xlabel("Popularity")
    plt.ylabel("Frequency")
    save_plot("matplotlib_4_popularity_distribution.png")

    scatter_data = df.dropna(subset=["danceability", "energy", "popularity"])
    scatter_sample = scatter_data.sample(
        n=min(2500, len(scatter_data)),
        random_state=42,
    )
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        scatter_sample["danceability"],
        scatter_sample["energy"],
        c=scatter_sample["popularity"],
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
    )
    plt.colorbar(scatter, label="Popularity")
    plt.title("Matplotlib 5: Danceability vs Energy", fontsize=14, weight="bold")
    plt.xlabel("Danceability")
    plt.ylabel("Energy")
    save_plot("matplotlib_5_danceability_vs_energy.png")


def create_seaborn_visuals(
    df: pd.DataFrame,
    genre_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
) -> None:
    """Create 5 visualizations using seaborn."""
    sns.set_theme(style="whitegrid", palette="deep")

    top_10_genres = df["genre"].value_counts().head(10).index.tolist()
    top_6_genres = df["genre"].value_counts().head(6).index.tolist()

    plt.figure(figsize=(12, 6))
    ax = sns.countplot(
        data=df[df["genre"].isin(top_10_genres)],
        x="genre",
        order=top_10_genres,
        hue="genre",
        palette="mako",
    )
    if ax.legend_:
        ax.legend_.remove()
    plt.title("Seaborn 1: Top 10 Genres by Track Count", fontsize=14, weight="bold")
    plt.xlabel("Genre")
    plt.ylabel("Track Count")
    plt.xticks(rotation=45, ha="right")
    save_plot("seaborn_1_top_genres_count.png")

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=genre_summary.head(10),
        x="genre",
        y="avg_popularity",
        hue="genre",
        palette="flare",
    )
    if ax.legend_:
        ax.legend_.remove()
    plt.title("Seaborn 2: Average Popularity by Genre", fontsize=14, weight="bold")
    plt.xlabel("Genre")
    plt.ylabel("Average Popularity")
    plt.xticks(rotation=45, ha="right")
    save_plot("seaborn_2_avg_popularity_by_genre.png")

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=df[df["genre"].isin(top_6_genres)],
        x="genre",
        y="popularity",
        hue="genre",
        dodge=False,
        palette="pastel",
    )
    if ax.legend_:
        ax.legend_.remove()
    plt.title("Seaborn 3: Popularity Spread of the Top 6 Genres", fontsize=14, weight="bold")
    plt.xlabel("Genre")
    plt.ylabel("Popularity")
    plt.xticks(rotation=30, ha="right")
    save_plot("seaborn_3_popularity_boxplot.png")

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=year_summary,
        x="release_year",
        y="avg_tempo",
        marker="o",
        linewidth=2.5,
        color="#264653",
    )
    plt.title("Seaborn 4: Average Tempo by Release Year (2015 Onward)", fontsize=14, weight="bold")
    plt.xlabel("Release Year")
    plt.ylabel("Average Tempo")
    save_plot("seaborn_4_avg_tempo_by_year.png")

    corr_columns = [
        "popularity",
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    correlation = df[corr_columns].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Seaborn 5: Correlation Heatmap of Audio Features", fontsize=14, weight="bold")
    save_plot("seaborn_5_correlation_heatmap.png")


def main() -> None:
    df = load_data()
    genre_summary, year_summary, rock_songs = build_summaries(df)

    print("=== Genre Popularity Summary ===")
    print(genre_summary.head(10).to_string(index=False))

    print("\n=== Release Year Tempo Summary ===")
    print(year_summary.to_string(index=False))

    print("\n=== Top Rock Songs by Popularity ===")
    print(rock_songs.to_string(index=False))

    create_matplotlib_visuals(df, genre_summary, year_summary, rock_songs)
    create_seaborn_visuals(df, genre_summary, year_summary)

    print(f"\nSaved 10 visualizations to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
