from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def run_eda(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Sentiment distribution
    sentiment_counts = df["sentiment"].value_counts()
    ax = sentiment_counts.plot(kind="bar", figsize=(7, 5))
    ax.set_title("IMDb Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(out / "sentiment_distribution.png", dpi=150)
    plt.close()

    # 2. Review length
    df = df.copy()
    df["word_count"] = df["clean_review"].str.split().str.len()
    plt.figure(figsize=(8, 5))
    for label in sorted(df["sentiment"].unique()):
        subset = df.loc[df["sentiment"] == label, "word_count"]
        plt.hist(subset, bins=30, alpha=0.55, label=label)
    plt.title("Review Length by Sentiment")
    plt.xlabel("Words per Review")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "review_length_by_sentiment.png", dpi=150)
    plt.close()

    # 3. Rating distribution if available
    if "rating" in df.columns:
        ratings = pd.to_numeric(df["rating"], errors="coerce").dropna()
        if not ratings.empty:
            plt.figure(figsize=(8, 5))
            plt.hist(ratings, bins=10)
            plt.title("IMDb Rating Distribution")
            plt.xlabel("Rating")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(out / "rating_distribution.png", dpi=150)
            plt.close()

    # 4. Genre trends if available
    if "genre" in df.columns:
        genre_counts = (
            df["genre"].dropna().astype(str).str.split(",").explode().str.strip()
            .value_counts().head(10)
        )
        if not genre_counts.empty:
            ax = genre_counts.sort_values().plot(kind="barh", figsize=(8, 5))
            ax.set_title("Top Genres in the Dataset")
            ax.set_xlabel("Number of Reviews")
            ax.set_ylabel("Genre")
            plt.tight_layout()
            plt.savefig(out / "top_genres.png", dpi=150)
            plt.close()

    print(f"EDA charts saved to: {out.resolve()}")
