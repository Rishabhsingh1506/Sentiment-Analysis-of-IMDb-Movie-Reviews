import re
from pathlib import Path
import pandas as pd

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_RE = re.compile(r"[^a-zA-Z\s]")


def clean_text(text: str) -> str:
    """Clean an IMDb review while preserving word-level information."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("&amp;", " and ")
    text = NON_LETTER_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(path: str, sample_size: int = 1000) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put your IMDb CSV at data/imdb_reviews_1000.csv"
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment' columns.")

    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df[df["sentiment"].isin(["positive", "negative"])]

    df["review"] = df["review"].astype(str)
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    df["clean_review"] = df["review"].apply(clean_text)
    df = df[df["clean_review"].str.len() > 0].reset_index(drop=True)

    return df
