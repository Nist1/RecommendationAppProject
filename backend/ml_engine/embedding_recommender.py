import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRecommender:
    """Рекомендации на основе sentence-transformers (семантические эмбеддинги)"""

    # Модель: paraphrase-multilingual-mpnet-base-v2 (sentence-transformers)
    # Поддерживает 50+ языков включая русский, размерность эмбеддингов 768
    # HuggingFace: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
    VECTORS_FILE = 'sbert_vectors.pkl'

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.model = None
        self.vectors = None
        self.df = None
        self._load()

    def _load(self):
        from sentence_transformers import SentenceTransformer

        print("Loading SBERT model...")
        self.model = SentenceTransformer(self.MODEL_NAME)

        vectors_path = os.path.join(self.models_dir, self.VECTORS_FILE)
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(
                f"SBERT vectors not found: {vectors_path}. "
                "Upload a dataset first to generate them."
            )

        with open(vectors_path, 'rb') as f:
            self.vectors = pickle.load(f)

        metadata_path = os.path.join(self.models_dir, 'api_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.df = pd.read_json(metadata_path)
        print(f"SBERT ready: {len(self.df)} docs, vector dim={self.vectors.shape[1]}")

    def get_recommendations_for_user(self, user_query: str, n: int = 5) -> pd.DataFrame:
        if not user_query or not user_query.strip():
            raise ValueError("Query cannot be empty")

        query_vector = self.model.encode([user_query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        top_indices = similarities.argsort()[-n:][::-1]

        results = self.df.iloc[top_indices].copy()
        results['similarity'] = similarities[top_indices]
        results = results.sort_values('similarity', ascending=False)
        return results[['vector_index', 'title', 'text', 'similarity']]


def build_sbert_vectors(models_dir: str):
    """Строит и сохраняет SBERT-векторы для уже загруженного датасета."""
    from sentence_transformers import SentenceTransformer

    metadata_path = os.path.join(models_dir, 'api_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Run TF-IDF pipeline first.")

    df = pd.read_json(metadata_path)
    texts = (df['title'].fillna('') + ' ' + df['text'].fillna('')).tolist()

    print(f"Encoding {len(texts)} documents with SBERT...")
    model = SentenceTransformer(EmbeddingRecommender.MODEL_NAME)
    vectors = model.encode(texts, show_progress_bar=True, batch_size=64)

    vectors_path = os.path.join(models_dir, EmbeddingRecommender.VECTORS_FILE)
    with open(vectors_path, 'wb') as f:
        pickle.dump(vectors, f)

    print(f"SBERT vectors saved: {vectors.shape}")
