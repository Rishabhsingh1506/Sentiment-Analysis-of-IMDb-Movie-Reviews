from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


def train_and_evaluate(df: pd.DataFrame, output_dir: str = "outputs"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"],
        df["sentiment"],
        test_size=0.20,
        random_state=42,
        stratify=df["sentiment"],
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(y_test, predictions, output_dict=True)
    metrics = {
        "accuracy": float(accuracy),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": int(X_train_tfidf.shape[1]),
    }

    with open(out / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(report).transpose().to_csv(
        out / "classification_report.csv"
    )

    cm = confusion_matrix(y_test, predictions, labels=["negative", "positive"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "positive"],
    )
    disp.plot()
    plt.title("IMDb Sentiment Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

    return model, vectorizer, accuracy
