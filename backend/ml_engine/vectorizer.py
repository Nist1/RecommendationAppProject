"""
Модуль векторизации текста
Включает: TF-IDF векторизацию, сохранение/загрузку моделей
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Union, List, Tuple
import os
from pathlib import Path


class TextVectorizer:
    """Класс для векторизации текста с использованием TF-IDF"""
    
    def __init__(self, max_features: int = 5000, 
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.8):
        """
        Инициализация векторизатора
        
        Args:
            max_features: максимальное количество признаков (слов)
            ngram_range: диапазон n-грамм (1,1) = только слова, (1,2) = слова + биграммы
            min_df: минимальная частота документов (игнорировать редкие слова)
            max_df: максимальная частота документов (игнорировать слишком частые слова)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,  # Логарифмическое масштабирование TF
            norm='l2'  # L2 нормализация
        )
        self.vectors = None
        self.feature_names = None
        
        print(f"Initialized TF-IDF Vectorizer:")
        print(f"  - Max features: {max_features}")
        print(f"  - N-gram range: {ngram_range}")
        print(f"  - Min document frequency: {min_df}")
        print(f"  - Max document frequency: {max_df}")
    
    def fit_transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Обучение векторизатора и трансформация текстов
        
        Args:
            texts: список или Series с текстами
            
        Returns:
            разреженная матрица векторов
        """
        print("\n" + "="*60)
        print("FITTING AND TRANSFORMING TEXTS")
        print("="*60)
        
        # Конвертация в список если нужно
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        print(f"Processing {len(texts)} documents...")
        
        # Обучение и трансформация
        self.vectors = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Статистика
        print(f"\nVectorization complete!")
        print(f"  - Vector shape: {self.vectors.shape}")
        print(f"  - Number of documents: {self.vectors.shape[0]}")
        print(f"  - Vocabulary size: {len(self.feature_names)}")
        print(f"  - Sparsity: {(1.0 - self.vectors.nnz / (self.vectors.shape[0] * self.vectors.shape[1])) * 100:.2f}%")
        print(f"  - Memory usage: {self.vectors.data.nbytes / 1024 / 1024:.2f} MB")
        
        return self.vectors
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Трансформация новых текстов (векторизатор уже обучен)
        
        Args:
            texts: список или Series с текстами
            
        Returns:
            разреженная матрица векторов
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        return self.vectorizer.transform(texts)
    
    def get_top_features(self, vector_index: int, n: int = 10) -> List[Tuple[str, float]]:
        """
        Получение топ-N признаков для конкретного документа
        
        Args:
            vector_index: индекс документа
            n: количество топ признаков
            
        Returns:
            список кортежей (признак, вес)
        """
        if self.vectors is None:
            raise ValueError("Vectors not computed yet. Run fit_transform first.")
        
        # Получение вектора документа
        doc_vector = self.vectors[vector_index].toarray().flatten()
        
        # Топ индексы
        top_indices = doc_vector.argsort()[-n:][::-1]
        
        # Топ признаки и их веса
        top_features = [(self.feature_names[i], doc_vector[i]) for i in top_indices]
        
        return top_features
    
    def get_vocabulary_stats(self) -> dict:
        """
        Получение статистики по словарю
        
        Returns:
            словарь со статистикой
        """
        if self.feature_names is None:
            raise ValueError("Vectorizer not fitted yet.")
        
        # IDF значения
        idf_values = self.vectorizer.idf_
        
        return {
            'vocabulary_size': len(self.feature_names),
            'mean_idf': np.mean(idf_values),
            'min_idf': np.min(idf_values),
            'max_idf': np.max(idf_values),
            'std_idf': np.std(idf_values)
        }
    
    def save_model(self, output_dir: str):
        """
        Сохранение векторизатора и векторов
        
        Args:
            output_dir: директория для сохранения
        """
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        # Создание директории если не существует
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
        vectors_path = os.path.join(output_dir, 'text_vectors.pkl')
        
        # Сохранение векторизатора
        print(f"Saving vectorizer to: {vectorizer_path}")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Сохранение векторов
        if self.vectors is not None:
            print(f"Saving vectors to: {vectors_path}")
            with open(vectors_path, 'wb') as f:
                pickle.dump(self.vectors, f)
        
        # Сохранение статистики
        stats_path = os.path.join(output_dir, 'vectorizer_stats.txt')
        print(f"Saving stats to: {stats_path}")
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("TF-IDF Vectorizer Statistics\n")
            f.write("="*50 + "\n\n")
            f.write(f"Vector shape: {self.vectors.shape}\n")
            f.write(f"Vocabulary size: {len(self.feature_names)}\n")
            
            stats = self.get_vocabulary_stats()
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nTop 20 features by IDF:\n")
            idf_scores = list(zip(self.feature_names, self.vectorizer.idf_))
            idf_scores.sort(key=lambda x: x[1], reverse=True)
            for word, score in idf_scores[:20]:
                f.write(f"  {word}: {score:.4f}\n")
        
        print("\nModel saved successfully!")
    
    def load_model(self, output_dir: str):
        """
        Загрузка векторизатора и векторов
        
        Args:
            output_dir: директория с моделями
        """
        print("\n" + "="*60)
        print("LOADING MODEL")
        print("="*60)
        
        vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
        vectors_path = os.path.join(output_dir, 'text_vectors.pkl')
        
        # Загрузка векторизатора
        print(f"Loading vectorizer from: {vectorizer_path}")
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Загрузка векторов
        if os.path.exists(vectors_path):
            print(f"Loading vectors from: {vectors_path}")
            with open(vectors_path, 'rb') as f:
                self.vectors = pickle.load(f)
            
            print(f"\nModel loaded successfully!")
            print(f"  - Vector shape: {self.vectors.shape}")
            print(f"  - Vocabulary size: {len(self.feature_names)}")
        else:
            print("Warning: Vectors file not found")


def vectorize_dataset(preprocessed_csv_path: str, 
                     output_dir: str,
                     text_column: str = 'combined_text',
                     max_features: int = 5000) -> Tuple[pd.DataFrame, TextVectorizer]:
    """
    Основная функция для векторизации датасета
    
    Args:
        preprocessed_csv_path: путь к обработанному CSV
        output_dir: директория для сохранения моделей
        text_column: название колонки с обработанным текстом
        max_features: максимальное количество признаков
        
    Returns:
        кортеж (DataFrame, TextVectorizer)
    """
    print("="*60)
    print("DATASET VECTORIZATION")
    print("="*60)
    
    # Загрузка обработанных данных
    print(f"\n1. Loading preprocessed data from: {preprocessed_csv_path}")
    df = pd.read_csv(preprocessed_csv_path)
    print(f"   Loaded {len(df)} records")
    
    # Проверка наличия колонки
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")
    
    # Фильтрация пустых текстов
    initial_count = len(df)
    df = df[df[text_column].notna() & (df[text_column].str.len() > 0)]
    print(f"   Filtered {initial_count - len(df)} empty texts")
    print(f"   Remaining: {len(df)} records")
    
    # Инициализация векторизатора
    print(f"\n2. Initializing vectorizer...")
    vectorizer = TextVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Униграммы и биграммы
        min_df=2,
        max_df=0.8
    )
    
    # Векторизация
    print(f"\n3. Vectorizing texts...")
    vectors = vectorizer.fit_transform(df[text_column])
    
    # Добавление информации о векторах в DataFrame
    df['vector_index'] = range(len(df))
    
    # Сохранение моделей
    print(f"\n4. Saving models...")
    vectorizer.save_model(output_dir)
    
    print("\n" + "="*60)
    print("VECTORIZATION COMPLETE!")
    print("="*60)
    
    return df, vectorizer


if __name__ == '__main__':
    # Пример использования
    df, vectorizer = vectorize_dataset(
        preprocessed_csv_path='data/processed/russian_news_processed.csv',
        output_dir='backend/ml_engine/models',
        text_column='combined_text',
        max_features=5000
    )
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Vocabulary size: {len(vectorizer.feature_names)}")