"""
Relevance filtering, sentiment scoring and annotation sampling for the
Kalshi pilot articles.

One-time install (in a terminal or a notebook cell):

    pip install vaderSentiment sentence-transformers transformers torch

Run from the notebook after collect_articles has produced articles_df:

    from kalshi_news_scoring import (
        score_relevance, score_sentiment, export_annotation_sample
    )

    scored_df = score_relevance(articles_df, revised_df)
    scored_df = score_sentiment(scored_df)
    annotation_df = export_annotation_sample(scored_df, revised_df)

Models download on first use (a few hundred MB). Every score is computed
once per distinct headline, so syndicated copies are not rescored.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data/pilot")

RELEVANCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINBERT_MODEL = "ProsusAI/finbert"


def _clean(text):
    """Strip markdown asterisks and GDELT's spaced punctuation."""
    text = str(text).replace("**", "")
    text = re.sub(r"\s+([,.:;!?%])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 1. Relevance
# ---------------------------------------------------------------------------

def contract_text(contracts_df):
    """
    Question text used for relevance. Multi-outcome markets put the
    outcome in yes_sub_title ("Who will win the LA mayoral election?" +
    "Karen Bass"), so it is appended when the title does not contain it.
    """
    df = contracts_df.set_index("ticker")
    text = df["title"].map(_clean)
    if "yes_sub_title" in df.columns:
        sub = df["yes_sub_title"].fillna("").map(_clean)
        add = [bool(b) and b.lower() not in t.lower()
               for t, b in zip(text, sub)]
        text = text.where(~pd.Series(add, index=text.index),
                          text + " " + sub)
    return text


def score_relevance(articles_df, contracts_df, threshold=0.35, batch_size=128,
                    embed_fn=None):
    """
    Cosine similarity between each headline and its contract's question.
    Adds relevance and is_relevant columns; nothing is dropped, so the
    threshold can be changed later without rescoring.
    """
    if embed_fn is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(RELEVANCE_MODEL)

        def embed_fn(texts):
            return model.encode(texts, batch_size=batch_size,
                                normalize_embeddings=True,
                                show_progress_bar=True)

    df = articles_df.copy()
    df["headline"] = df["title"].map(_clean)

    questions = (contract_text(contracts_df)
                 .reindex(df["ticker"].unique()))

    headlines = df["headline"].unique().tolist()
    h_vec = dict(zip(headlines, embed_fn(headlines)))
    q_vec = dict(zip(questions.index, embed_fn(questions.tolist())))

    df["relevance"] = [
        float(np.dot(h_vec[h], q_vec[t]))
        for h, t in zip(df["headline"], df["ticker"])
    ]
    df["is_relevant"] = df["relevance"] >= threshold

    summary = (df.groupby("ticker")
               .agg(articles=("url", "count"),
                    relevant=("is_relevant", "sum"),
                    median_relevance=("relevance", "median"))
               .reset_index())
    summary["pct_relevant"] = (summary["relevant"] / summary["articles"]
                               * 100).round(1)
    print(f"Relevance threshold {threshold}: "
          f"{df['is_relevant'].sum()} of {len(df)} articles kept")
    display(summary)
    return df


def show_relevance_examples(scored_df, ticker, n=8, threshold=0.35):
    """Highest and lowest scoring headlines for one contract, for choosing a threshold."""
    sub = (scored_df[scored_df["ticker"] == ticker]
           .drop_duplicates("headline")
           .sort_values("relevance", ascending=False))
    cols = ["relevance", "headline"]
    print("Most relevant:")
    display(sub[cols].head(n))
    print("Least relevant:")
    display(sub[cols].tail(n))
    print("Around the threshold:")
    mid = sub.iloc[(sub["relevance"] - threshold).abs().argsort()[:n]]
    display(mid[cols].sort_values("relevance", ascending=False))


# ---------------------------------------------------------------------------
# 2. Sentiment
# ---------------------------------------------------------------------------

def score_sentiment(scored_df, batch_size=32, vader_fn=None, finbert_fn=None,
                    reuse=True):
    """
    Add VADER (neg/neu/pos/compound) and FinBERT probabilities per headline.
    With reuse=True, headlines already scored in pilot_articles_scored.csv
    keep their saved scores and only new headlines are run through the models.
    """
    df = scored_df.drop(columns=[c for c in scored_df.columns
                                 if c.startswith(("vader_", "finbert_"))])
    all_headlines = df["headline"].unique().tolist()

    saved = pd.DataFrame()
    saved_path = OUT_DIR / "pilot_articles_scored.csv"
    if reuse and saved_path.exists():
        cols = pd.read_csv(saved_path, nrows=0).columns
        keep = ["headline"] + [c for c in cols
                               if c.startswith(("vader_", "finbert_"))]
        if len(keep) > 1:
            saved = (pd.read_csv(saved_path, usecols=keep)
                     .drop_duplicates("headline"))
            saved = saved[saved["headline"].isin(all_headlines)]
    done = set(saved["headline"]) if len(saved) else set()
    headlines = [h for h in all_headlines if h not in done]
    print(f"{len(done)} headlines reuse saved scores, "
          f"{len(headlines)} new headlines to score")

    if not headlines:
        df = df.merge(saved, on="headline", how="left")
        df.to_csv(saved_path, index=False)
        print("Saved", saved_path)
        return df

    if vader_fn is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        vader_fn = analyzer.polarity_scores

    if finbert_fn is None:
        from transformers import pipeline
        clf = pipeline("text-classification", model=FINBERT_MODEL,
                       top_k=None, truncation=True)

        def finbert_fn(texts):
            return clf(texts, batch_size=batch_size)

    vader = pd.DataFrame([vader_fn(h) for h in headlines])
    vader.columns = [f"vader_{c}" for c in vader.columns]
    vader["headline"] = headlines

    finbert_rows = []
    for start in range(0, len(headlines), 512):
        chunk = headlines[start:start + 512]
        for h, scores in zip(chunk, finbert_fn(chunk)):
            row = {"headline": h}
            row.update({f"finbert_{s['label'].lower()}": s["score"]
                        for s in scores})
            finbert_rows.append(row)
        print(f"  FinBERT {min(start + 512, len(headlines))}/{len(headlines)}")
    finbert = pd.DataFrame(finbert_rows)
    finbert["finbert_net"] = (finbert["finbert_positive"]
                              - finbert["finbert_negative"])

    new_scores = vader.merge(finbert, on="headline")
    scores = pd.concat([saved, new_scores], ignore_index=True)
    df = df.merge(scores, on="headline", how="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "pilot_articles_scored.csv", index=False)
    print("Saved", OUT_DIR / "pilot_articles_scored.csv")
    return df


# ---------------------------------------------------------------------------
# 3. Annotation sample for stance labels
# ---------------------------------------------------------------------------

def export_annotation_sample(scored_df, contracts_df, per_contract=12,
                             seed=42):
    """
    Stratified sample of relevant, distinct stories for two annotators.

    Each row pairs a headline with its contract question. Annotators fill
    annotator_1 and annotator_2 with YES (makes the YES outcome more likely),
    NO (makes it less likely) or NEUTRAL (relevant but no direction, or
    irrelevant). Half the sample per contract comes from the top half of
    relevance scores and half from the bottom half, so the labels also
    check the relevance threshold.
    """
    questions = contract_text(contracts_df)

    pool = (scored_df[scored_df["is_relevant"]]
            .drop_duplicates(["ticker", "cluster_id"]))

    samples = []
    for ticker, group in pool.groupby("ticker"):
        median = group["relevance"].median()
        high = group[group["relevance"] >= median]
        low = group[group["relevance"] < median]
        k = per_contract // 2
        samples.append(high.sample(min(k, len(high)), random_state=seed))
        samples.append(low.sample(min(per_contract - k, len(low)),
                                  random_state=seed))

    sample = pd.concat(samples).sample(frac=1, random_state=seed)
    sample = sample.assign(
        question=sample["ticker"].map(questions),
        annotator_1="",
        annotator_2="",
        notes="",
    )[["ticker", "question", "published", "domain", "headline", "url",
       "relevance", "annotator_1", "annotator_2", "notes"]]
    sample = sample.reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "stance_annotation_sample.csv"
    sample.to_csv(path, index=False)
    print(f"Saved {len(sample)} rows for annotation to {path}")
    return sample
