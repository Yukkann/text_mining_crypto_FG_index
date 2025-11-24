import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

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
    # print(f"✔ 已生成清理後 CSV：{output_csv}")
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
# KNN 模型：文字情緒分類
# -----------------------------------

def run_knn_from_labels(df, text_col="clean_text", label_col="labels", k=5):
    """
    df: DataFrame，裡面已經有 clean_text 和 labels 欄
    labels 欄內容為 'g' / 'b' / 'n'，未標記者為 NaN 或空字串
    """

    # 1. 取出有標籤的資料當 training set
    labeled_mask = df[label_col].notna() & (df[label_col] != "")
    df_labeled = df[labeled_mask]

    if df_labeled.empty:
        print("⚠ 沒有任何有標籤的資料（labels 欄全是空的），KNN 無法訓練。")
        return df

    train_texts = df_labeled[text_col].astype(str).tolist()
    train_labels = df_labeled[label_col].astype(str).tolist()

    # 2. 全部文字（要一起轉成向量，方便之後一次 predict）
    all_texts = df[text_col].astype(str).tolist()

    # 3. TF-IDF 向量化（只用文字，不用 upvotes / comments）
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95
    )
    X_all = vectorizer.fit_transform(all_texts)
    X_train = X_all[df_labeled.index]  # 用有標籤的那幾列當訓練資料

    # 4. 建立並訓練 KNN 模型
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='cosine'   # 若版本不支援，可以改成 'minkowski'（預設歐式距離）
    )
    knn.fit(X_train, train_labels)

    # 5. 對「全部資料」做預測（包含原本有標籤 & 沒標籤）
    knn_pred = knn.predict(X_all)

    # 6. 寫回 DataFrame：新增一欄 'knn_sentiment'
    df["knn_sentiment"] = knn_pred

    print(f"✔ KNN 已訓練完成，並寫入欄位 'knn_sentiment'（k = {k}）")
    return df


if __name__ == "__main__":

    # Step 1：生成清理後 CSV（這裡會產生 clean_text，也還是算出 vader sentiment 和 influence）
    input_csv = "ETH_2024-10_2025-09.csv"
    output_csv = "ETH_with_sentiment.csv"
    df = process_reddit_csv(input_csv, output_csv)

    # Step 2：用 labels 欄跑 KNN（g/b/n）
    df = run_knn_from_labels(df, text_col="clean_text", label_col="labels", k=5)

    # Step 3：把含 KNN 結果的整份 CSV 存起來
    df.to_csv("ETH_with_sentiment_and_KNN.csv", index=False)
    # print("✔ 已輸出：ETH_with_sentiment_and_KNN.csv")
    # Step 4：將 KNN 預測結果寫入 CSV
    df.to_csv("ETH_with_KNN_sentiment.csv", index=False)
    # print("✔ 已輸出 KNN 結果到：ETH_with_KNN_sentiment.csv")

    # Step 5: 詞頻/LDA/WordCloud 都可繼續用
    show_word_frequency(df)
    show_sentiment_keywords(df)
    run_lda(df)
    top_words = extract_tfidf_keywords(df)
    print("\n🔹 Top TF-IDF Keywords:")
    for w, s in top_words:
        print(f"{w}: {s:.4f}")
    print("\n🔹 Top TF-IDF Keywords:")
    for w, s in top_words:
        print(f"{w}: {s:.4f}")
    # Step 6：WordCloud
    plot_wordcloud(df[df["sentiment"] > 0]["clean_text"], "Positive WordCloud")
    plot_wordcloud(df[df["sentiment"] < 0]["clean_text"], "Negative WordCloud")
