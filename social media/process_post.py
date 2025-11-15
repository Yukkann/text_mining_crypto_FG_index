import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from collections import Counter
import nltk

# -----------------------------------
# 文字清理函數
# -----------------------------------
def clean_post(text):
    if pd.isna(text):
        return ""
    
    text = text.lower() # 小寫
    text = re.sub(r'http\S+|www.\S+', '', text) # 移除網址
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # 移除 emoji
    text = re.sub(r'[^a-z0-9\s]', ' ', text) # 移除標點符號
    text = re.sub(r'\s+', ' ', text).strip() # 移除多餘空白

    # 預設英文停用字
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
    # Tokenization
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    # 移除過短字
    words = [w for w in words if len(w) > 1]

    return ' '.join(words)

# -----------------------------------
# 產生清理 + sentiment + influence CSV
# -----------------------------------
def process_reddit_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['clean_text'] = df['full_text'].apply(clean_post)

    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['clean_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    df['influence'] = df['sentiment'] * np.log1p(df['upvotes'] + df['num_comments'])

    df.to_csv(output_csv, index=False)
    print(f"✔ 已生成清理後 CSV：{output_csv}")
    return df


# -----------------------------------
# 詞頻分析
# -----------------------------------
def show_word_frequency(df, top_k=30):
    all_words = " ".join(df["clean_text"]).split()
    freq = Counter(all_words).most_common(top_k)
    print(f"\n🔹 Top {top_k} Word Frequency:")
    print(freq)


# -----------------------------------
# 正負向詞頻
# -----------------------------------
def show_sentiment_keywords(df):
    pos_posts = df[df["sentiment"] > 0]["clean_text"]
    neg_posts = df[df["sentiment"] < 0]["clean_text"]

    pos_words = Counter(" ".join(pos_posts).split()).most_common(20)
    neg_words = Counter(" ".join(neg_posts).split()).most_common(20)

    print("\n正向關鍵詞：", pos_words)
    print("負向關鍵詞：", neg_words)


# -----------------------------------
# LDA 主題模型
# -----------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def run_lda(df, column="clean_text", num_topics=5, num_words=12):
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[column])
    terms = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=num_topics,
        learning_method='batch',
        random_state=42
    )
    lda.fit(X)

    for idx, topic in enumerate(lda.components_):
        print(f"\n🟦 Topic {idx}")
        top_indices = topic.argsort()[-num_words:][::-1]
        top_words = [terms[i] for i in top_indices]
        print(top_words)


# -----------------------------------
# TF-IDF 排序
# -----------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_keywords(df, column="clean_text", top_k=20):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df[column])
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = X.mean(axis=0).A1
    top_indices = tfidf_scores.argsort()[-top_k:][::-1]

    return [(feature_names[i], tfidf_scores[i]) for i in top_indices]


# -----------------------------------
# WordCloud
# -----------------------------------
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_wordcloud(text_list, title):
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text_list))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


# -----------------------------------
# 主程式
# -----------------------------------
if __name__ == "__main__":

    # Step 1：生成清理後 CSV
    input_csv = "ETH_2024-10_2025-09.csv"
    output_csv = "ETH_with_sentiment.csv"
    df = process_reddit_csv(input_csv, output_csv)

    # Step 2：詞頻
    show_word_frequency(df)

    # Step 3：正負關鍵詞
    show_sentiment_keywords(df)

    # Step 4：LDA 主題模型
    run_lda(df)

    # Step 5：TF-IDF
    top_words = extract_tfidf_keywords(df)
    print("\n🔹 Top TF-IDF Keywords:")
    for w, s in top_words:
        print(f"{w}: {s:.4f}")

    # Step 6：WordCloud
    plot_wordcloud(df[df["sentiment"] > 0]["clean_text"], "Positive WordCloud")
    plot_wordcloud(df[df["sentiment"] < 0]["clean_text"], "Negative WordCloud")