"""
Merged Sentiment Analysis Pipeline
Combines text preprocessing from process_post.py with three model approaches from sentiment_comparison
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# Text preprocessing imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Model imports
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

# Analysis imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configurations
FINBERT_MODEL = "ProsusAI/finbert"
REDDIT_CRYPTO_MODEL = "mwkby/distilbert-base-uncased-sentiment-reddit-crypto"
CRYPTOBERT_MODEL = "ElKulako/cryptobert"

# Model parameters
MAX_LENGTH = 512
DEVICE = -1  # Force CPU (use 0 or -1 for CPU, or device number for GPU)

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# TEXT PREPROCESSING (from process_post.py)
# ============================================================================

def clean_post(text):
    """
    Clean and preprocess text for sentiment analysis.
    - Converts to lowercase
    - Removes URLs and emojis
    - Removes special characters
    - Removes stopwords
    - Applies lemmatization
    """
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove emojis
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    stop_words = set(stopwords.words('english'))
    # 自訂停用字
    custom_stopwords = {
        # 與主題太相關、太常見的字
        "eth", "ethereum", "crypto", "btc", "bitcoin", "price", "market", "time",
        "day", "week", "month", "year", "today", "tomorrow", "news", "update", "token", "blockchain",
        "network", "chain", "project", "community", "development", "decentralized", "defi", "nft",
        "smart", "contract", "platform", "platforms",

        # 無意義功能字
        "would", "could", "think", "also", "know", "really", "like",
        "get", "got", "even", "one", "make", "made", "much", "many",
        "still", "see", "say", "said", "way", "every", "bit", "lot", "use", 
        "thing", "people", "going", "since", "may", "everyone", "something",
    }

    # 合併兩份停用字
    stop_words = stop_words.union(custom_stopwords)
    
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    words = [w for w in words if len(w) > 1]  # Remove single character words

    return ' '.join(words)


# ============================================================================
# VADER SENTIMENT ANALYSIS (from process_post.py)
# ============================================================================

def analyze_vader_sentiment(df, text_column='clean_text'):
    """
    Analyze sentiment using VADER (baseline method from process_post.py).
    """
    analyzer = SentimentIntensityAnalyzer()
    df['vader_sentiment'] = df[text_column].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )
    return df


# ============================================================================
# TRANSFORMER MODEL FUNCTIONS (from sentiment_comparison)
# ============================================================================

def chunk_text(
    text: str,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_LENGTH,
    stride: int = 50,
) -> List[str]:
    """Split text into overlapping chunks for long posts."""
    if not text or not text.strip():
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunk_size = max_length - 2  # reserve space for special tokens
    if chunk_size <= 0:
        chunk_size = max_length

    step = max(1, chunk_size - stride)
    chunks: List[str] = []

    for start in range(0, len(token_ids), step):
        end = start + chunk_size
        ids_slice = token_ids[start:end]
        if not ids_slice:
            break
        chunks.append(tokenizer.decode(ids_slice, skip_special_tokens=True))
        if end >= len(token_ids):
            break

    return chunks or [tokenizer.decode(token_ids, skip_special_tokens=True)]


def score_text(
    text: str,
    pipeline: TextClassificationPipeline,
    tokenizer: AutoTokenizer,
    label_map: Dict[str, str],
    max_length: int = MAX_LENGTH,
) -> Dict[str, float]:
    """Average sentiment probabilities across chunks and normalize labels."""
    chunks = chunk_text(text, tokenizer, max_length=max_length)
    if not chunks:
        return {
            "label": "neutral",
            "confidence": 0.33,
            "prob_negative": 0.33,
            "prob_neutral": 0.34,
            "prob_positive": 0.33,
            "sentiment_score": 0.0,
        }

    outputs = pipeline(chunks, truncation=True, max_length=max_length)
    if isinstance(outputs, dict):
        outputs = [outputs]

    # Aggregate scores across chunks
    score_dict = {"negative": [], "neutral": [], "positive": []}
    
    for result in outputs:
        if isinstance(result, list):
            # return_all_scores=True
            for entry in result:
                mapped_label = label_map.get(entry["label"].lower(), entry["label"].lower())
                if mapped_label in score_dict:
                    score_dict[mapped_label].append(entry["score"])
        else:
            # Single prediction
            mapped_label = label_map.get(result["label"].lower(), result["label"].lower())
            if mapped_label in score_dict:
                score_dict[mapped_label].append(result["score"])

    # Average scores
    avg_scores = {
        k: np.mean(v) if v else 0.0 
        for k, v in score_dict.items()
    }
    
    # Find best label
    best_label = max(avg_scores, key=avg_scores.get)
    
    return {
        "label": best_label,
        "confidence": float(avg_scores[best_label]),
        "prob_negative": float(avg_scores["negative"]),
        "prob_neutral": float(avg_scores["neutral"]),
        "prob_positive": float(avg_scores["positive"]),
        "sentiment_score": float(avg_scores["positive"] - avg_scores["negative"]),
    }


def load_model_pipeline(model_name: str, device: int) -> Tuple[TextClassificationPipeline, Dict[str, str]]:
    """Load a sentiment model and return pipeline + label mapping."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    pipeline = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        top_k=None,  # Return all scores
        device=device,
    )
    
    # Label mapping to standardize outputs to positive/negative/neutral
    label_map = {}
    if "cryptobert" in model_name.lower():
        # CryptoBERT uses: Bearish, Neutral, Bullish
        label_map = {"bearish": "negative", "neutral": "neutral", "bullish": "positive"}
    elif "finbert" in model_name.lower():
        # FinBERT uses: positive, negative, neutral
        label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
    else:
        # Reddit crypto model and others use standard labels
        label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
    
    return pipeline, label_map


def run_finbert_analysis(df, text_column='combined_text'):
    """Run FinBERT sentiment analysis."""
    print(f"\n{'='*80}")
    print("Running FinBERT Analysis")
    print(f"{'='*80}")
    
    finbert_pipeline, finbert_label_map = load_model_pipeline(FINBERT_MODEL, DEVICE)
    finbert_tokenizer = finbert_pipeline.tokenizer

    print("\nAnalyzing sentiment with FinBERT...")
    finbert_results: List[Dict[str, float]] = []

    for text in tqdm(df[text_column], desc="FinBERT", total=len(df)):
        result = score_text(text, finbert_pipeline, finbert_tokenizer, finbert_label_map, max_length=MAX_LENGTH)
        finbert_results.append(result)

    # Add results to dataframe
    df["finbert_label"] = [r["label"] for r in finbert_results]
    df["finbert_confidence"] = [r["confidence"] for r in finbert_results]
    df["finbert_prob_neg"] = [r["prob_negative"] for r in finbert_results]
    df["finbert_prob_neu"] = [r["prob_neutral"] for r in finbert_results]
    df["finbert_prob_pos"] = [r["prob_positive"] for r in finbert_results]
    df["finbert_sentiment_score"] = [r["sentiment_score"] for r in finbert_results]

    print("\nFinBERT sentiment distribution:")
    print(df["finbert_label"].value_counts())
    print(f"\nAverage confidence: {df['finbert_confidence'].mean():.4f}")
    print(f"Average sentiment score: {df['finbert_sentiment_score'].mean():.4f} (range: -1 to +1)")
    
    return df


def run_reddit_crypto_analysis(df, text_column='combined_text'):
    """Run Reddit Crypto Model sentiment analysis."""
    print(f"\n{'='*80}")
    print("Running Reddit Crypto Model Analysis")
    print(f"{'='*80}")
    
    reddit_pipeline, reddit_label_map = load_model_pipeline(REDDIT_CRYPTO_MODEL, DEVICE)
    reddit_tokenizer = reddit_pipeline.tokenizer

    print("\nAnalyzing sentiment with Reddit Crypto Model...")
    reddit_results: List[Dict[str, float]] = []

    for text in tqdm(df[text_column], desc="Reddit Crypto", total=len(df)):
        result = score_text(text, reddit_pipeline, reddit_tokenizer, reddit_label_map, max_length=MAX_LENGTH)
        reddit_results.append(result)

    # Add results to dataframe
    df["reddit_label"] = [r["label"] for r in reddit_results]
    df["reddit_confidence"] = [r["confidence"] for r in reddit_results]
    df["reddit_prob_neg"] = [r["prob_negative"] for r in reddit_results]
    df["reddit_prob_neu"] = [r["prob_neutral"] for r in reddit_results]
    df["reddit_prob_pos"] = [r["prob_positive"] for r in reddit_results]
    df["reddit_sentiment_score"] = [r["sentiment_score"] for r in reddit_results]

    print("\nReddit Crypto sentiment distribution:")
    print(df["reddit_label"].value_counts())
    print(f"\nAverage confidence: {df['reddit_confidence'].mean():.4f}")
    print(f"Average sentiment score: {df['reddit_sentiment_score'].mean():.4f} (range: -1 to +1)")
    
    return df


def run_cryptobert_analysis(df, text_column='combined_text'):
    """Run CryptoBERT sentiment analysis."""
    print(f"\n{'='*80}")
    print("Running CryptoBERT Analysis")
    print(f"{'='*80}")
    
    cryptobert_pipeline, cryptobert_label_map = load_model_pipeline(CRYPTOBERT_MODEL, DEVICE)
    cryptobert_tokenizer = cryptobert_pipeline.tokenizer

    print("\nAnalyzing sentiment with CryptoBERT...")
    cryptobert_results: List[Dict[str, float]] = []

    for text in tqdm(df[text_column], desc="CryptoBERT", total=len(df)):
        result = score_text(text, cryptobert_pipeline, cryptobert_tokenizer, cryptobert_label_map, max_length=MAX_LENGTH)
        cryptobert_results.append(result)

    # Add results to dataframe
    df["cryptobert_label"] = [r["label"] for r in cryptobert_results]
    df["cryptobert_confidence"] = [r["confidence"] for r in cryptobert_results]
    df["cryptobert_prob_neg"] = [r["prob_negative"] for r in cryptobert_results]
    df["cryptobert_prob_neu"] = [r["prob_neutral"] for r in cryptobert_results]
    df["cryptobert_prob_pos"] = [r["prob_positive"] for r in cryptobert_results]
    df["cryptobert_sentiment_score"] = [r["sentiment_score"] for r in cryptobert_results]

    print("\nCryptoBERT sentiment distribution:")
    print(df["cryptobert_label"].value_counts())
    print(f"\nAverage confidence: {df['cryptobert_confidence'].mean():.4f}")
    print(f"Average sentiment score: {df['cryptobert_sentiment_score'].mean():.4f} (range: -1 to +1)")
    
    return df


# ============================================================================
# TEXT ANALYSIS FUNCTIONS (from process_post.py)
# ============================================================================

def show_word_frequency(df, column="clean_text", top_k=30):
    """Show top K word frequencies."""
    all_words = " ".join(df[column]).split()
    freq = Counter(all_words).most_common(top_k)
    print(f"\n🔹 Top {top_k} Word Frequency:")
    for word, count in freq:
        print(f"  {word}: {count}")
    return freq


def show_sentiment_keywords(df, sentiment_column="vader_sentiment", text_column="clean_text"):
    """Show positive and negative keywords based on sentiment."""
    pos_posts = df[df[sentiment_column] > 0][text_column]
    neg_posts = df[df[sentiment_column] < 0][text_column]

    pos_words = Counter(" ".join(pos_posts).split()).most_common(20)
    neg_words = Counter(" ".join(neg_posts).split()).most_common(20)

    print("\n正向關鍵詞 (Positive Keywords):")
    for word, count in pos_words:
        print(f"  {word}: {count}")
    
    print("\n負向關鍵詞 (Negative Keywords):")
    for word, count in neg_words:
        print(f"  {word}: {count}")
    
    return pos_words, neg_words


def run_lda(df, column="clean_text", num_topics=5, num_words=12):
    """Run LDA topic modeling."""
    print(f"\n🔹 Running LDA Topic Modeling ({num_topics} topics)...")
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[column])
    terms = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=num_topics,
        learning_method='batch',
        random_state=42
    )
    lda.fit(X)

    topics = []
    for idx, topic in enumerate(lda.components_):
        print(f"\n🟦 Topic {idx}")
        top_indices = topic.argsort()[-num_words:][::-1]
        top_words = [terms[i] for i in top_indices]
        print(f"  {', '.join(top_words)}")
        topics.append(top_words)
    
    return topics


def extract_tfidf_keywords(df, column="clean_text", top_k=20):
    """Extract top TF-IDF keywords."""
    print(f"\n🔹 Extracting Top {top_k} TF-IDF Keywords...")
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df[column])
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = X.mean(axis=0).A1
    top_indices = tfidf_scores.argsort()[-top_k:][::-1]

    keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices]
    print("\nTop TF-IDF Keywords:")
    for w, s in keywords:
        print(f"  {w}: {s:.4f}")
    
    return keywords


def plot_wordcloud(text_list, title, figsize=(10, 5)):
    """Plot word cloud."""
    if not text_list or len(text_list) == 0:
        print(f"No text data for {title}")
        return
    
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text_list))
    plt.figure(figsize=figsize)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================================
# COMPARISON AND ANALYSIS FUNCTIONS
# ============================================================================

def analyze_model_agreement(df):
    """Analyze agreement between different sentiment models."""
    print(f"\n{'='*80}")
    print("MODEL AGREEMENT ANALYSIS")
    print(f"{'='*80}")
    
    # Calculate agreement between models
    if 'finbert_label' in df.columns and 'reddit_label' in df.columns:
        df["finbert_reddit_agree"] = df["finbert_label"] == df["reddit_label"]
        print(f"FinBERT ↔ Reddit Crypto: {df['finbert_reddit_agree'].mean():.2%}")
    
    if 'finbert_label' in df.columns and 'cryptobert_label' in df.columns:
        df["finbert_cryptobert_agree"] = df["finbert_label"] == df["cryptobert_label"]
        print(f"FinBERT ↔ CryptoBERT: {df['finbert_cryptobert_agree'].mean():.2%}")
    
    if 'reddit_label' in df.columns and 'cryptobert_label' in df.columns:
        df["reddit_cryptobert_agree"] = df["reddit_label"] == df["cryptobert_label"]
        print(f"Reddit Crypto ↔ CryptoBERT: {df['reddit_cryptobert_agree'].mean():.2%}")
    
    # All three models agree
    if all(col in df.columns for col in ['finbert_label', 'reddit_label', 'cryptobert_label']):
        df["all_agree"] = (
            (df["finbert_label"] == df["reddit_label"]) & 
            (df["reddit_label"] == df["cryptobert_label"])
        )
        print(f"All 3 models agree: {df['all_agree'].mean():.2%} ({df['all_agree'].sum()} posts)")
    
    return df


def plot_sentiment_comparison(df):
    """Plot sentiment distribution comparison across models."""
    print(f"\n{'='*80}")
    print("SENTIMENT DISTRIBUTION COMPARISON")
    print(f"{'='*80}")
    
    models = []
    if 'finbert_label' in df.columns:
        models.append(('FinBERT', 'finbert_label'))
    if 'reddit_label' in df.columns:
        models.append(('Reddit Crypto', 'reddit_label'))
    if 'cryptobert_label' in df.columns:
        models.append(('CryptoBERT', 'cryptobert_label'))
    
    if not models:
        print("No model results found for comparison.")
        return
    
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
    
    for idx, (model_name, col_name) in enumerate(models):
        counts = df[col_name].value_counts()
        model_colors = [colors.get(label, "#3498db") for label in counts.index]
        axes[idx].bar(counts.index, counts.values, color=model_colors, alpha=0.8, edgecolor="black")
        axes[idx].set_title(model_name, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Sentiment", fontsize=10)
        axes[idx].set_ylabel("Count", fontsize=10)
        axes[idx].grid(axis="y", alpha=0.3)
        
        for i, (label, count) in enumerate(counts.items()):
            axes[idx].text(i, count + 5, f"{count}\n({count/len(df)*100:.1f}%)", 
                          ha="center", va="bottom", fontweight="bold")
    
    plt.tight_layout()
    plt.show()


def plot_confidence_comparison(df):
    """Plot confidence score comparison across models."""
    print(f"\n{'='*80}")
    print("CONFIDENCE SCORE COMPARISON")
    print(f"{'='*80}")
    
    confidence_cols = {}
    if 'finbert_confidence' in df.columns:
        confidence_cols['FinBERT'] = df['finbert_confidence']
    if 'reddit_confidence' in df.columns:
        confidence_cols['Reddit Crypto'] = df['reddit_confidence']
    if 'cryptobert_confidence' in df.columns:
        confidence_cols['CryptoBERT'] = df['cryptobert_confidence']
    
    if not confidence_cols:
        print("No confidence scores found.")
        return
    
    # Print statistics
    for model_name, conf_scores in confidence_cols.items():
        print(f"\n{model_name}:")
        print(f"  Mean: {conf_scores.mean():.4f}")
        print(f"  Median: {conf_scores.median():.4f}")
        print(f"  Std: {conf_scores.std():.4f}")
    
    # Box plot
    confidence_data = pd.DataFrame(confidence_cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    confidence_data.boxplot(ax=ax, patch_artist=True)
    ax.set_title("Confidence Score Distribution by Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Confidence Score", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sentiment_scores_comparison(df):
    """Plot continuous sentiment score distributions."""
    print(f"\n{'='*80}")
    print("CONTINUOUS SENTIMENT SCORE COMPARISON")
    print(f"{'='*80}")
    
    score_cols = {}
    if 'finbert_sentiment_score' in df.columns:
        score_cols['FinBERT'] = df['finbert_sentiment_score']
    if 'reddit_sentiment_score' in df.columns:
        score_cols['Reddit Crypto'] = df['reddit_sentiment_score']
    if 'cryptobert_sentiment_score' in df.columns:
        score_cols['CryptoBERT'] = df['cryptobert_sentiment_score']
    if 'vader_sentiment' in df.columns:
        score_cols['VADER'] = df['vader_sentiment']
    
    if not score_cols:
        print("No sentiment scores found.")
        return
    
    # Print statistics
    for model_name, scores in score_cols.items():
        print(f"\n{model_name}:")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Median: {scores.median():.4f}")
        print(f"  Std: {scores.std():.4f}")
        print(f"  Min: {scores.min():.4f}")
        print(f"  Max: {scores.max():.4f}")
    
    # Histogram comparison
    n_models = len(score_cols)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    colors_list = ["#3498db", "#e67e22", "#9b59b6", "#e74c3c"]
    
    for idx, (model_name, scores) in enumerate(score_cols.items()):
        axes[idx].hist(scores, bins=30, color=colors_list[idx % len(colors_list)], 
                      alpha=0.7, edgecolor="black")
        axes[idx].axvline(x=0, color="red", linestyle="--", linewidth=2, label="Neutral (0)")
        axes[idx].axvline(x=scores.mean(), color="green", linestyle="--", linewidth=2, 
                         label=f"Mean ({scores.mean():.3f})")
        axes[idx].set_title(f"{model_name} Sentiment Score", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Sentiment Score (-1 to +1)", fontsize=10)
        axes[idx].set_ylabel("Frequency", fontsize=10)
        axes[idx].legend()
        axes[idx].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_reddit_csv(input_csv, output_csv=None, run_models=True, run_analysis=True):
    """
    Main processing pipeline that combines text preprocessing and sentiment analysis.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file
    output_csv : str, optional
        Path to output CSV file (if None, auto-generates name)
    run_models : bool
        Whether to run transformer models (can be slow)
    run_analysis : bool
        Whether to run text analysis (word frequency, LDA, etc.)
    """
    print(f"\n{'='*80}")
    print("MERGED SENTIMENT ANALYSIS PIPELINE")
    print(f"{'='*80}")
    
    # Load data
    print(f"\nLoading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} posts")
    
    # Combine title and text
    df['title'] = df['title'].fillna('').astype(str)
    df['text'] = df['text'].fillna('').astype(str)
    df['full_text'] = df['title'] + ' ' + df['text']
    df['combined_text'] = (df['title'].str.strip() + '. ' + df['text'].str.strip()).str.strip()
    df['combined_text'] = df['combined_text'].str.replace(r'^\. ', '', regex=True)
    
    # Convert date if available
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
    
    # Step 1: Text Preprocessing
    print(f"\n{'='*80}")
    print("STEP 1: TEXT PREPROCESSING")
    print(f"{'='*80}")
    df['clean_text'] = df['full_text'].apply(clean_post)
    print(f"Text preprocessing completed. Average cleaned text length: {df['clean_text'].str.len().mean():.0f} characters")
    
    # Step 2: VADER Sentiment Analysis (baseline)
    print(f"\n{'='*80}")
    print("STEP 2: VADER SENTIMENT ANALYSIS (Baseline)")
    print(f"{'='*80}")
    df = analyze_vader_sentiment(df, text_column='clean_text')
    df['influence'] = df['vader_sentiment'] * np.log1p(df['upvotes'] + df['num_comments'])
    print(f"VADER sentiment analysis completed.")
    print(f"Average VADER sentiment: {df['vader_sentiment'].mean():.4f}")
    
    # Step 3: Transformer Model Analysis (if requested)
    if run_models:
        print(f"\n{'='*80}")
        print("STEP 3: TRANSFORMER MODEL ANALYSIS")
        print(f"{'='*80}")
        print("This may take a while...")
        
        # Run all three models (using cleaned text)
        df = run_finbert_analysis(df, text_column='clean_text')
        df = run_reddit_crypto_analysis(df, text_column='clean_text')
        df = run_cryptobert_analysis(df, text_column='clean_text')
        
        # Model agreement analysis
        df = analyze_model_agreement(df)
        
        # Visualizations
        plot_sentiment_comparison(df)
        plot_confidence_comparison(df)
        plot_sentiment_scores_comparison(df)
    
    # Step 4: Text Analysis (if requested)
    if run_analysis:
        print(f"\n{'='*80}")
        print("STEP 4: TEXT ANALYSIS")
        print(f"{'='*80}")
        
        # Word frequency
        show_word_frequency(df, column='clean_text', top_k=30)
        
        # Sentiment keywords
        show_sentiment_keywords(df, sentiment_column='vader_sentiment', text_column='clean_text')
        
        # LDA topic modeling
        run_lda(df, column='clean_text', num_topics=5, num_words=12)
        
        # TF-IDF keywords
        extract_tfidf_keywords(df, column='clean_text', top_k=20)
        
        # WordClouds
        pos_texts = df[df['vader_sentiment'] > 0.2]['clean_text'].tolist()
        neg_texts = df[df['vader_sentiment'] < -0.2]['clean_text'].tolist()
        if pos_texts:
            plot_wordcloud(pos_texts, "Positive Sentiment WordCloud")
        if neg_texts:
            plot_wordcloud(neg_texts, "Negative Sentiment WordCloud")
    
    # Save results
    if output_csv:
        # Drop temporary columns if needed
        df_output = df.copy()
        if 'combined_text' in df_output.columns:
            # Keep combined_text for reference, or drop if preferred
            pass
        
        df_output.to_csv(output_csv, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to {output_csv}")
        print(f"{'='*80}")
    
    # Summary statistics
    print_summary(df)
    
    return df


def print_summary(df):
    """Print summary statistics."""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nDataset: {len(df):,} posts")
    if 'created_at' in df.columns:
        print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    
    # VADER summary
    if 'vader_sentiment' in df.columns:
        print(f"\nVADER Sentiment:")
        print(f"  Mean: {df['vader_sentiment'].mean():.4f}")
        print(f"  Median: {df['vader_sentiment'].median():.4f}")
        print(f"  Std: {df['vader_sentiment'].std():.4f}")
    
    # Model summaries
    models_summary = []
    if 'finbert_label' in df.columns:
        models_summary.append(('FinBERT', 'finbert_label', 'finbert_sentiment_score'))
    if 'reddit_label' in df.columns:
        models_summary.append(('Reddit Crypto', 'reddit_label', 'reddit_sentiment_score'))
    if 'cryptobert_label' in df.columns:
        models_summary.append(('CryptoBERT', 'cryptobert_label', 'cryptobert_sentiment_score'))
    
    for model_name, label_col, score_col in models_summary:
        print(f"\n{model_name}:")
        print(f"  Distribution:")
        for sentiment in ["positive", "neutral", "negative"]:
            count = (df[label_col] == sentiment).sum()
            pct = count / len(df) * 100
            print(f"    {sentiment.capitalize()}: {count:,} ({pct:.1f}%)")
        if score_col in df.columns:
            print(f"  Sentiment Score:")
            print(f"    Mean: {df[score_col].mean():.4f}")
            print(f"    Median: {df[score_col].median():.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    CWD = Path.cwd()
    DATA_DIR = CWD if CWD.name == "social media" else CWD / "social media"
    INPUT_CSV = DATA_DIR / "ETH_2024-10_2025-09.csv"
    OUTPUT_CSV = DATA_DIR / "ETH_merged_sentiment_analysis.csv"
    
    # Run the pipeline
    # Set run_models=False to skip transformer models (faster for testing)
    # Set run_analysis=False to skip text analysis
    df = process_reddit_csv(
        input_csv=str(INPUT_CSV),
        output_csv=str(OUTPUT_CSV),
        run_models=True,  # Set to False to skip transformer models
        run_analysis=True  # Set to False to skip text analysis
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

