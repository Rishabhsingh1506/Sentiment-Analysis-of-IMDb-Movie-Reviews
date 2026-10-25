from pathlib import Path
import pandas as pd

from preprocess import load_and_clean
from eda import run_eda
from train_model import train_and_evaluate

DATA_PATH = "data/imdb_reviews_1000.csv"
OUTPUT_PATH = "outputs"


def main():
    print("=" * 60)
    print("IMDb MOVIE REVIEW SENTIMENT ANALYSIS")
    print("=" * 60)

    df = load_and_clean(DATA_PATH, sample_size=1000)

    print(f"Final review count: {len(df)}")
    print("\nSentiment counts:")
    print(df["sentiment"].value_counts())

    cleaned_path = Path(OUTPUT_PATH) / "cleaned_imdb_reviews.csv"
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    df.to_csv(cleaned_path, index=False)

    run_eda(df, OUTPUT_PATH)
    train_and_evaluate(df, OUTPUT_PATH)

    print("\nProject completed successfully.")
    print(f"Results are available in: {OUTPUT_PATH}/")


if __name__ == "__main__":
    main()
