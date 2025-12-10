"""
Модуль рекомендаций на основе контента
- Косинусное сходство
- Поиск N наиболее похожих документов
- Обработка ошибок
"""

import numpy as np
import pandas as pd
import sys
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_engine.vectorizer import TextVectorizer
from ml_engine.preprocessor import RussianTextPreprocessor


class ContentRecommender:
    """Контентные рекомендации на основе TF-IDF и косинусного сходства"""
    
    def __init__(self, models_dir: str, df: pd.DataFrame = None):
        """
        Args:
            models_dir: путь к директории с моделями
            df: DataFrame с данными (опционально, загрузится автоматически)
        """
        self.models_dir = models_dir
        self.vectorizer = None
        self.vectors = None
        self.df = df
        self.preprocessor = RussianTextPreprocessor()
        
        # Автоматическая загрузка моделей
        self._load_models()
        
        # Загрузка метаданных если не передан df
        if self.df is None:
            self._load_metadata()
    
    def _load_models(self):
        """Загрузка векторизатора и векторов"""
        print("Loading models...")
        
        vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
        vectors_path = os.path.join(self.models_dir, 'text_vectors.pkl')
        
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
        
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f"Vectors not found: {vectors_path}")
        
        # Загрузка vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Загрузка vectors
        with open(vectors_path, 'rb') as f:
            self.vectors = pickle.load(f)
        
        print(f"✓ Models loaded successfully")
        print(f"  Vector shape: {self.vectors.shape}")
    
    def _load_metadata(self):
        """Загрузка метаданных из JSON"""
        metadata_path = os.path.join(self.models_dir, 'api_metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.df = pd.read_json(metadata_path)
        print(f"✓ Metadata loaded: {len(self.df)} records")
    
    def preprocess_query(self, query: str) -> str:
        """
        Предобработка пользовательского запроса
        
        Args:
            query: пользовательский запрос
            
        Returns:
            обработанный запрос
        """
        return self.preprocessor.preprocess(query)
    
    def get_similar_items(self, item_id: int, n: int = 5):
        """
        Поиск N наиболее похожих документов по ID
        
        Args:
            item_id: ID документа (vector_index)
            n: количество похожих документов
            
        Returns:
            DataFrame с похожими документами
        """
        # Проверка существования ID
        if 'vector_index' not in self.df.columns:
            raise ValueError("DataFrame must have 'vector_index' column")
        
        if item_id not in self.df['vector_index'].values:
            raise ValueError(f"Item with vector_index={item_id} not found")
        
        # Получение индекса в DataFrame
        idx = self.df[self.df['vector_index'] == item_id].index[0]
        
        # Получение вектора документа
        doc_vector = self.vectors[idx:idx+1]
        
        # Вычисление косинусного сходства
        similarities = cosine_similarity(doc_vector, self.vectors).flatten()
        
        # Исключаем сам документ
        similarities[idx] = -1
        
        # Топ-N похожих
        top_indices = similarities.argsort()[-n:][::-1]
        
        # Формирование результата
        results = self.df.iloc[top_indices].copy()
        results['similarity'] = similarities[top_indices]
        
        # Сортировка по убыванию similarity
        results = results.sort_values('similarity', ascending=False)
        
        return results[['vector_index', 'title', 'text', 'similarity']]
    
    def get_recommendations_for_user(self, user_query: str, n: int = 5):
        """
        Поиск N наиболее релевантных документов по запросу пользователя
        
        Args:
            user_query: текстовый запрос пользователя
            n: количество рекомендаций
            
        Returns:
            DataFrame с рекомендациями
        """
        if not user_query or not user_query.strip():
            raise ValueError("Query cannot be empty")
        
        # Предобработка запроса
        processed_query = self.preprocess_query(user_query)
        
        if not processed_query:
            raise ValueError("Query preprocessing resulted in empty string")
        
        # Векторизация запроса
        query_vector = self.vectorizer.transform([processed_query])
        
        # Вычисление косинусного сходства
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Топ-N релевантных
        top_indices = similarities.argsort()[-n:][::-1]
        
        # Формирование результата
        results = self.df.iloc[top_indices].copy()
        results['similarity'] = similarities[top_indices]
        
        # Сортировка по убыванию similarity
        results = results.sort_values('similarity', ascending=False)
        
        return results[['vector_index', 'title', 'text', 'similarity']]
    
    def get_statistics(self) -> dict:
        """Получение статистики рекомендательной системы"""
        return {
            'total_documents': len(self.df),
            'vocabulary_size': len(self.vectorizer.get_feature_names_out()),
            'vector_dimensions': self.vectors.shape[1],
            'sparsity': (1.0 - self.vectors.nnz / (self.vectors.shape[0] * self.vectors.shape[1])) * 100
        }


def main():
    """Пример использования"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Content-based Recommender System')
    parser.add_argument('--models-dir', type=str, default='backend/ml_engine/models',
                       help='Directory with models')
    parser.add_argument('--query', type=str, default='крушение самолета',
                       help='User query for recommendations')
    parser.add_argument('--item-id', type=int, default=None,
                       help='Item ID for similar items')
    parser.add_argument('--n', type=int, default=5,
                       help='Number of recommendations')
    
    args = parser.parse_args()
    
    try:
        # Инициализация рекомендательной системы
        print("\n" + "="*60)
        print("INITIALIZING RECOMMENDER SYSTEM")
        print("="*60)
        
        recommender = ContentRecommender(models_dir=args.models_dir)
        
        # Статистика
        stats = recommender.get_statistics()
        print(f"\nSystem Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Рекомендации по запросу
        if args.query:
            print("\n" + "="*60)
            print("QUERY-BASED RECOMMENDATIONS")
            print("="*60)
            print(f"\nUser query: '{args.query}'")
            
            recommendations = recommender.get_recommendations_for_user(args.query, n=args.n)
            
            print(f"\nTop {args.n} recommendations:")
            for i, row in recommendations.iterrows():
                print(f"\n{i+1}. [ID: {row['vector_index']}] Similarity: {row['similarity']:.4f}")
                print(f"   Title: {row['title'][:80]}...")
                print(f"   Text: {row['text'][:150]}...")
        
        # Похожие документы
        if args.item_id is not None:
            print("\n" + "="*60)
            print("SIMILAR ITEMS")
            print("="*60)
            print(f"\nFinding items similar to ID: {args.item_id}")
            
            similar = recommender.get_similar_items(args.item_id, n=args.n)
            
            print(f"\nTop {args.n} similar items:")
            for i, row in similar.iterrows():
                print(f"\n{i+1}. [ID: {row['vector_index']}] Similarity: {row['similarity']:.4f}")
                print(f"   Title: {row['title'][:80]}...")
                print(f"   Text: {row['text'][:150]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()