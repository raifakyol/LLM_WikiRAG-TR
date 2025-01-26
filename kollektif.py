import nltk
import json
nltk.download('punkt', quiet=True)
import numpy as np
import seaborn as sns
import faiss
import torch
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from collections import Counter

# Hugging Face'den WikiRAG-TR veri kümesini yükleme
dataset = load_dataset("Metin/WikiRAG-TR")

# Dataset'i listeye dönüştür
train_data = list(dataset["train"])

# Rastgele 1000 soru seçme 
random.seed(42)
sample_size = 1000
sampled_data = random.sample(train_data, sample_size)

# Soruların kelime uzunluklarını hesaplama
question_lengths = [len(nltk.word_tokenize(sample['question'])) for sample in sampled_data]

# Soruların kelime uzunluk dağılımını görselleştirilmesi
plt.figure(figsize=(10, 6))
plt.hist(question_lengths, bins=20, edgecolor='black', alpha=0.7)
plt.title('Soruların Kelime Uzunluk Dağılımı')
plt.xlabel('Kelime Sayısı')
plt.ylabel('Soru Sayısı')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Her soru için 5 chunk oluşturma
# Sorulara karşılık gelen metinleri bölümlere ayırma..
sampled_chunks = []
sampled_true_chunks = []
sampled_true_indices = []
invalid_indices = []

for example in sampled_data:
    try:
        # Bölünme noktalarını alma.
        ctx_split_points = json.loads(example.get("ctx_split_points", "[]"))
    except json.JSONDecodeError:
        ctx_split_points = []

    # Tam metin ve bölümlerini oluşturma
    full_context = example.get("context", "")
    chunk_list = []

    start = 0
    for split_point in ctx_split_points:
        chunk_list.append(full_context[start:split_point])
        start = split_point

    # Eksik bölümler varsa boş string ile doldurma
    while len(chunk_list) < 5:
        chunk_list.append("")

    sampled_chunks.extend(chunk_list[:5])

    # Doğru bölüm indekslerini belirleme
    correct_idx = example.get("correct_intro_idx", -1)
    if 0 <= correct_idx < len(chunk_list):
        sampled_true_chunks.append(chunk_list[correct_idx])
        sampled_true_indices.append(len(sampled_chunks) - 5 + correct_idx)
    else:
        sampled_true_chunks.append("")
        sampled_true_indices.append(-1)
        invalid_indices.append({
            "correct_intro_idx": correct_idx,
            "chunk_list_length": len(chunk_list)
        })

#word matcing için.
tokenized_corpus = [word_tokenize(chunk) for chunk in sampled_chunks]


# BERT tokenizer ile BM25 için tokenize edilmiş corpus hazırlama
# BM25 modeli için BERT tokenizasyonunu kullanılır.
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  

# BERT tokenizer ile tokenized corpus oluşturma
bert_tokenized_corpus = [
    bert_tokenizer.tokenize(chunk) for chunk in sampled_chunks
]
bm25_bert = BM25Okapi(bert_tokenized_corpus)

# Word Matching için Jaccard Similarity fonksiyonu
# Jaccard benzerliği hesaplayan bir fonksiyon.
def jaccard_similarity(query, document):
    query_set = set(query)
    document_set = set(document)
    intersection = query_set.intersection(document_set)
    union = query_set.union(document_set)
    return len(intersection) / len(union) if len(union) > 0 else 0

# Word Matching Retrieval Yöntemi
# Sorgular için Word Matching ile eşleşen en iyi sonuçları getirir.
def word_matching_retrieval(query, top_k=5):
    tokenized_query = word_tokenize(query)
    scores = [jaccard_similarity(tokenized_query, doc) for doc in tokenized_corpus]
    return np.argsort(scores)[::-1][:top_k]

# Modeller ve Tokenizer'lar
model_names = [
    "sentence-transformers/all-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large-instruct",
    "ytu-ce-cosmos/turkish-colbert",
    "thenlper/gte-large",
    "BAAI/bge-m3",
]

models = []
tokenizers = []
vector_dims = []
indexes = []


# Modelleri yükleme ve FAISS için indeks oluşturma
# Modellerin yüklenmesi ve FAISS indeks yapısının kurulması.
for model_name in model_names:
    try:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        vector_dim = model.config.hidden_size
        index = faiss.IndexFlatL2(vector_dim)

        tokenizers.append(tokenizer)
        models.append(model)
        vector_dims.append(vector_dim)
        indexes.append(index)
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")

# Dense vektör üretme fonksiyonu
# Metinlerden vektör oluşturur.
def encode_chunks_with_model(chunks, model, tokenizer):
    inputs = tokenizer(chunks, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


# Türkçe özel dönüşüm fonksiyonu
def preprocess_text_turkish(text):
    return text.replace("I", "ı").lower()

# Chunk işleme fonksiyonu
def process_model_chunks(model, tokenizer, index, model_name):
    print(f"Processing model: {model_name} started.")
    batch_size = 64
    for start in range(0, len(sampled_chunks), batch_size):
        end = start + batch_size
        chunk_batch = sampled_chunks[start:end]
        vectors = encode_chunks_with_model(chunk_batch, model, tokenizer)
        index.add(vectors.astype("float32"))
    print(f"Model {model_name} processing completed. Total vectors in FAISS: {index.ntotal}")

# Turkish-ColBERT modeli için özel chunk işleme fonksiyonu
def process_model_chunks_turkish_colbert(model, tokenizer, index, model_name):
    print(f"Processing model (Turkish-ColBERT): {model_name} started.")
    batch_size = 64
    for start in range(0, len(sampled_chunks), batch_size):
        end = start + batch_size
        chunk_batch = [preprocess_text_turkish(chunk) for chunk in sampled_chunks[start:end]]
        vectors = encode_chunks_with_model(chunk_batch, model, tokenizer)
        index.add(vectors.astype("float32"))
    print(f"Model {model_name} processing completed. Total vectors in FAISS: {index.ntotal}")

processing_times = []
for model, tokenizer, index, model_name in zip(models, tokenizers, indexes, model_names):
    start_time = time.time()
    if model_name == "ytu-ce-cosmos/turkish-colbert":
        process_model_chunks_turkish_colbert(model, tokenizer, index, model_name)
    else:
        process_model_chunks(model, tokenizer, index, model_name)
    processing_time = time.time() - start_time
    processing_times.append(processing_time)

# Model bazlı işlem süresi grafiği
plt.figure(figsize=(10, 6))
plt.bar(model_names, processing_times, color="purple", alpha=0.7)
plt.title("Model Bazlı Chunk İşlem Süresi")
plt.xlabel("Model Names")
plt.ylabel("Time (seconds)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Retrieval yöntemleri
# BM25 ile bilgi alma

# BM25 bilgi alma fonksiyonu (BERT tokenizasyonu ile)
def bm25_retrieval(query, top_k=5):
    tokenized_query = bert_tokenizer.tokenize(query)  # Sorguyu BERT ile tokenize et
    scores = bm25_bert.get_scores(tokenized_query)   # BM25 skorlarını al
    return np.argsort(scores)[::-1][:top_k]          # En yüksek skora sahip ilk K sonucu döndür

def dense_retrieval(query, model, tokenizer, index, top_k=5):
    query_vector = encode_chunks_with_model([query], model, tokenizer)
    D, I = index.search(query_vector, top_k)
    return D[0], I[0]

# Skorları ağırlıklarla birleştirerek bilgi alma.
def weighted_combination_retrieval(query, bm25_scores, word_scores, dense_scores_list, weights, top_k=5):
    """
    Doğrusal Kombinasyon yöntemi ile skorları birleştirerek en iyi sonuçları döndürür.
    """
    # Ağırlıkları unpack et
    w_bm25, w_word, w_dense_list = weights[0], weights[1], weights[2:]

    # BM25 ve Word skorlarını normalize et
    normalized_bm25 = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores)) if np.max(bm25_scores) > np.min(bm25_scores) else bm25_scores
    normalized_word = (word_scores - np.min(word_scores)) / (np.max(word_scores) - np.min(word_scores)) if np.max(word_scores) > np.min(word_scores) else word_scores

    # Dense skorlarını normalize et ve ağırlıklı toplamı hesapla
    combined_dense = np.zeros_like(normalized_bm25)
    for dense_scores, w_dense in zip(dense_scores_list, w_dense_list):
        normalized_dense = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores)) if np.max(dense_scores) > np.min(dense_scores) else dense_scores
        combined_dense += w_dense * normalized_dense

    # Ağırlıklı toplamı hesapla
    combined_scores = w_bm25 * normalized_bm25 + w_word * normalized_word + combined_dense

    # En yüksek skora sahip Top-K sonuçları döndür
    return np.argsort(combined_scores)[::-1][:top_k]

# Performans değerlendirme fonksiyonu
def evaluate_retrieval(questions, true_indices, retrieval_func):
    top_1_count = 0
    top_5_count = 0
    for i, (q, true_index) in enumerate(zip(questions, true_indices)):
        retrieved_indices = retrieval_func(q)
        if true_index in retrieved_indices[:1]:
            top_1_count += 1
        if true_index in retrieved_indices[:5]:
            top_5_count += 1
    return {
        "Top-1 Accuracy": top_1_count / len(questions),
        "Top-5 Accuracy": top_5_count / len(questions),
    }

# Tüm yöntemlerin performansını ölçme
results = {}

# BM25 performans değerlendirmesi
results["BM25"] = evaluate_retrieval(
    [d["question"] for d in sampled_data], 
    sampled_true_indices, 
    lambda q: bm25_retrieval(q)
)

# Word Matching performans değerlendirmesi
results["Word"] = evaluate_retrieval(
    [d["question"] for d in sampled_data], 
    sampled_true_indices, 
    lambda q: word_matching_retrieval(q)
)

# Dense Retrieval performans değerlendirmesi
for model, tokenizer, index, model_name in zip(models, tokenizers, indexes, model_names):
    results[model_name] = evaluate_retrieval(
        [d["question"] for d in sampled_data], 
        sampled_true_indices, 
        lambda q: dense_retrieval(q, model, tokenizer, index)[1]
    )

# Ağırlıklı kombinasyon yöntemi için performans değerlendirme
weights = [0.1, 0.05, 0.05, 0.3, 0.05, 0.15, 0.3]  # BM25, Word Matching, Dense1, Dense2, Dense3, Dense4, Dense5 ağırlıkları

def combined_retrieval(query):
    bm25_scores = bm25_bert.get_scores(bert_tokenizer(query))
    word_scores = [jaccard_similarity(word_tokenize(query), doc) for doc in tokenized_corpus]
    dense_scores_list = [
        dense_retrieval(query, model, tokenizer, index, top_k=len(sampled_chunks))[0]
        for model, tokenizer, index in zip(models, tokenizers, indexes)
    ]
    return weighted_combination_retrieval(query, bm25_scores, word_scores, dense_scores_list, weights)

results["Weighted Combination"] = evaluate_retrieval(
    [d["question"] for d in sampled_data], 
    sampled_true_indices, 
    combined_retrieval
)

def majority_voting_retrieval(query, bm25_results, word_results, dense_results_list, top_k=5):
    combined_results = bm25_results + word_results
    for dense_results in dense_results_list:
        combined_results += list(dense_results)

    # Top-K sonuçları döndür (en çok oyu alanlar)
    result_counter = Counter(combined_results)
    top_results = result_counter.most_common(top_k)
    return [result[0] for result in top_results]

# Rank Averaging

def rank_averaging_retrieval(query, bm25_scores, word_scores, dense_scores_list, top_k=5):
    # Sıralamaları normalize et
    bm25_ranks = np.argsort(np.argsort(-bm25_scores)).astype(np.float64)  # Float olarak başlat
    word_scores = np.array(word_scores)
    word_ranks = np.argsort(np.argsort(-word_scores)).astype(np.float64)

    # Dense sıralamaları genişlet
    dense_ranks_list = []
    for dense_scores in dense_scores_list:
        dense_ranks_full = np.zeros_like(bm25_scores, dtype=np.float64)  # Float türünde sıfırlarla başla
        top_k_indices = np.argsort(-dense_scores)[:top_k]  # İlk top_k sıralamayı al
        for rank, idx in enumerate(top_k_indices):
            dense_ranks_full[idx] = rank  # Top-k sıralamaları doldur
        dense_ranks_list.append(dense_ranks_full)

    # Ortalama sıralama hesapla
    avg_ranks = bm25_ranks + word_ranks
    for dense_ranks in dense_ranks_list:
        avg_ranks += dense_ranks
    avg_ranks = avg_ranks / (2 + len(dense_ranks_list))

    # En yüksek sıralamaya sahip Top-K sonuçları döndür
    return np.argsort(avg_ranks)[:top_k]




# Geometric Mean

def geometric_mean_retrieval(query, bm25_scores, word_scores, dense_scores_list, top_k=5):
    # Skorları normalize et
    bm25_scores = np.clip(bm25_scores, 1e-8, None)
    word_scores = np.clip(word_scores, 1e-8, None)

    # Dense skorlarını genişlet
    expanded_dense_scores_list = []
    for dense_scores in dense_scores_list:
        expanded_dense_scores = np.ones_like(bm25_scores) * 1e-8  # Varsayılan olarak küçük bir değer
        top_k_indices = np.argsort(-dense_scores)[:top_k]
        for idx in top_k_indices:
            expanded_dense_scores[idx] = dense_scores[idx]
        expanded_dense_scores_list.append(expanded_dense_scores)

    # Geometrik ortalama hesapla
    combined_scores = bm25_scores * word_scores
    for dense_scores in expanded_dense_scores_list:
        combined_scores *= dense_scores

    combined_scores **= 1 / (2 + len(expanded_dense_scores_list))  # Geometrik ortalama

    # En yüksek skora sahip Top-K sonuçları döndür
    return np.argsort(combined_scores)[::-1][:top_k]



# Ensemble yöntemlerini değerlendirme
results["Majority Voting"] = evaluate_retrieval(
    [d["question"] for d in sampled_data],
    sampled_true_indices,
    lambda q: majority_voting_retrieval(
        q,
        bm25_retrieval(q),
        word_matching_retrieval(q),
        [dense_retrieval(q, model, tokenizer, index)[1] for model, tokenizer, index in zip(models, tokenizers, indexes)]
    )
)

results["Rank Averaging"] = evaluate_retrieval(
    [d["question"] for d in sampled_data],
    sampled_true_indices,
    lambda q: rank_averaging_retrieval(
        q,
        bm25_bert.get_scores(bert_tokenizer.tokenize(q)),
        [jaccard_similarity(word_tokenize(q), doc) for doc in tokenized_corpus],
        [dense_retrieval(q, model, tokenizer, index)[0] for model, tokenizer, index in zip(models, tokenizers, indexes)]
    )
)


results["Geometric Mean"] = evaluate_retrieval(
    [d["question"] for d in sampled_data],
    sampled_true_indices,
    lambda q: geometric_mean_retrieval(
        q,
        bm25_bert.get_scores(bert_tokenizer.tokenize(q)),
        [jaccard_similarity(word_tokenize(q), doc) for doc in tokenized_corpus],
        [dense_retrieval(q, model, tokenizer, index)[0] for model, tokenizer, index in zip(models, tokenizers, indexes)]
    )
)

# Sonuçları DataFrame'e dönüştürme
results_df = pd.DataFrame([
    {"Method": method, "Top-1 Accuracy": metrics["Top-1 Accuracy"], "Top-5 Accuracy": metrics["Top-5 Accuracy"]}
    for method, metrics in results.items()
])

# Sonuçları yazdır 
print("\nRetrieval Performance Results:")
print(results_df)

# Top-1 Accuracy
plt.figure(figsize=(10, 6))
plt.barh(results_df["Method"], results_df["Top-1 Accuracy"])
plt.title("Top-1 Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Methods")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# Top-5 Accuracy
plt.figure(figsize=(10, 6))
plt.barh(results_df["Method"], results_df["Top-5 Accuracy"])
plt.title("Top-5 Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Methods")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# Top-1 and Top-5 Accuracy
plt.figure(figsize=(12, 8))
plt.plot(results_df["Method"], results_df["Top-1 Accuracy"], label="Top-1 Accuracy", marker='o')
plt.plot(results_df["Method"], results_df["Top-5 Accuracy"], label="Top-5 Accuracy", marker='s')
plt.title("Top-1 and Top-5 Accuracy")
plt.xlabel("Methods")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.legend()
plt.grid(linestyle="--", alpha=0.7)
plt.show()

# Heatmap Görselleştirme
plt.figure(figsize=(10, 6))
sns.heatmap(
    results_df.set_index("Method"),
    annot=True,
    fmt=".2f",
    cmap="YlGnBu"
)
plt.title("Accuracy Heatmap")
plt.show()
