"""
Модуль предобработки текста для русского языка
Включает: очистку, токенизацию, удаление стоп-слов, лемматизацию
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List
import pymorphy3
import nltk
from nltk.corpus import stopwords

# Скачать стоп-слова при первом запуске
try:
    stopwords.words('russian')
except LookupError:
    nltk.download('stopwords')

class RussianTextPreprocessor:
    """Класс для предобработки русского текста"""
    
    def __init__(self):
        """Инициализация препроцессора"""
        print("Initializing Russian Text Preprocessor...")
        
        # Лемматизатор для русского языка
        self.morph = pymorphy3.MorphAnalyzer()
        
        # Стоп-слова для русского
        self.stop_words = set(stopwords.words('russian'))
        
        # Добавление дополнительных стоп-слов
        additional_stops = {
            'это', 'который', 'которая', 'которые', 'весь', 'свой',
            'мочь', 'год', 'также', 'другой', 'наш', 'ваш', 'их'
        }
        self.stop_words.update(additional_stops)
        
        # Пунктуация
        self.punctuation = string.punctuation + '«»—–'
        
        print(f"Loaded {len(self.stop_words)} stop words")
        print("Preprocessor ready!")
    
    def clean_text(self, text: str) -> str:
        """
        Базовая очистка текста
        
        Args:
            text: исходный текст
            
        Returns:
            очищенный текст
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление URL
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Удаление HTML тегов
        text = re.sub(r'<.*?>', '', text)
        
        # Удаление множественных пробелов и переносов строк
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление пунктуации
        text = text.translate(str.maketrans('', '', self.punctuation))
        
        # Удаление цифр (опционально - можно закомментировать)
        text = re.sub(r'\d+', '', text)
        
        # Удаление лишних пробелов
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Токенизация текста (разбиение на слова)
        
        Args:
            text: текст для токенизации
            
        Returns:
            список токенов
        """
        # Простая токенизация по пробелам
        tokens = text.split()
        
        # Фильтрация токенов (минимум 2 символа)
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Удаление стоп-слов
        Args:
            tokens: список токенов
            
        Returns:
            список токенов без стоп-слов
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Лемматизация токенов (приведение к начальной форме)
        Args:
            tokens: список токенов
            
        Returns:
            список лемматизированных токенов
        """
        lemmas = []
        for token in tokens:
            # Получение нормальной формы слова
            parsed = self.morph.parse(token)[0]
            lemma = parsed.normal_form
            lemmas.append(lemma)
        
        return lemmas
    
    def preprocess(self, text: str, remove_stops: bool = True, 
                   lemmatize: bool = True) -> str:
        """
        Полный пайплайн предобработки текста
        
        Args:
            text: исходный текст
            remove_stops: удалять ли стоп-слова
            lemmatize: применять ли лемматизацию
            
        Returns:
            обработанный текст
        """
        # Очистка
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        # Токенизация
        tokens = self.tokenize(text)
        
        # Удаление стоп-слов
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Лемматизация
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        # Объединение обратно в строку
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str], batch_size: int = 100,
                        remove_stops: bool = True, lemmatize: bool = True) -> List[str]:
        """
        Пакетная обработка текстов с индикатором прогресса
        
        Args:
            texts: список текстов
            batch_size: размер батча для вывода прогресса
            remove_stops: удалять ли стоп-слова
            lemmatize: применять ли лемматизацию
            
        Returns:
            список обработанных текстов
        """
        processed_texts = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            processed = self.preprocess(text, remove_stops, lemmatize)
            processed_texts.append(processed)
            
            # Вывод прогресса
            if (i + 1) % batch_size == 0:
                print(f"Processed {i + 1}/{total} texts ({(i+1)/total*100:.1f}%)")
        
        print(f"Completed! Processed {total} texts")
        return processed_texts
    
    def get_text_stats(self, text: str) -> dict:
        """
        Получение статистики по тексту
        
        Args:
            text: текст для анализа
            
        Returns:
            словарь со статистикой
        """
        tokens = self.tokenize(self.clean_text(text))
        
        return {
            'char_count': len(text),
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'avg_word_length': np.mean([len(w) for w in tokens]) if tokens else 0
        }


def process_russian_news_dataset(input_path: str, output_path: str,
                                 text_columns: List[str] = ['title', 'text'],
                                 sample_size: int = None) -> pd.DataFrame:
    """
    Основная функция для обработки датасета русских новостей
    
    Args:
        input_path: путь к исходному CSV файлу
        output_path: путь для сохранения обработанного CSV
        text_columns: список колонок с текстом для обработки
        sample_size: количество записей для обработки (None = все)
        
    Returns:
        обработанный DataFrame
    """
    print("="*60)
    print("RUSSIAN NEWS DATASET PREPROCESSING")
    print("="*60)
    
    # Загрузка данных
    print(f"\n1. Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
    
    # Выборка (если нужна)
    if sample_size and sample_size < len(df):
        print(f"\n   Taking sample of {sample_size} records")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Проверка наличия колонок
    missing_cols = [col for col in text_columns if col not in df.columns]
    if missing_cols:
        print(f"\n   WARNING: Missing columns: {missing_cols}")
        text_columns = [col for col in text_columns if col in df.columns]
    
    print(f"\n   Will process columns: {text_columns}")
    
    # Удаление дубликатов
    print(f"\n2. Removing duplicates...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=text_columns, keep='first')
    print(f"   Removed {initial_count - len(df)} duplicates")
    print(f"   Remaining: {len(df)} records")
    
    # Удаление пустых значений
    print(f"\n3. Removing empty values...")
    df = df.dropna(subset=text_columns)
    print(f"   Remaining: {len(df)} records")
    
    # Инициализация препроцессора
    print(f"\n4. Initializing preprocessor...")
    preprocessor = RussianTextPreprocessor()
    
    # Обработка каждой текстовой колонки
    print(f"\n5. Processing text columns...")
    for col in text_columns:
        print(f"\n   Processing '{col}' column:")
        
        # Статистика до обработки
        sample_text = df[col].iloc[0]
        before_stats = preprocessor.get_text_stats(sample_text)
        print(f"   Sample before: {sample_text[:100]}...")
        print(f"   Stats before: {before_stats}")
        
        # Обработка
        df[f'{col}_processed'] = preprocessor.preprocess_batch(
            df[col].tolist(),
            batch_size=100
        )
    
        # Статистика после обработки
        sample_processed = df[f'{col}_processed'].iloc[0]
        after_stats = preprocessor.get_text_stats(sample_processed)
        print(f"   Sample after: {sample_processed[:100]}...")
        print(f"   Stats after: {after_stats}")
    
    # Создание объединенного текста
    print(f"\n6. Creating combined text field...")
    processed_cols = [f'{col}_processed' for col in text_columns]
    df['combined_text'] = df[processed_cols].apply(
        lambda x: ' '.join(x.astype(str)), axis=1
    )
    
    # Удаление пустых обработанных текстов
    initial_count = len(df)
    df = df[df['combined_text'].str.len() > 10]
    print(f"   Removed {initial_count - len(df)} records with too short text")
    
    # Сброс индекса
    df = df.reset_index(drop=True)
    
    # Добавление ID если нет
    if 'id' not in df.columns:
        df['id'] = range(len(df))

    # Сохранение результата
    print(f"\n7. Saving processed data to: {output_path}")
    cols_to_save = ['id', 'title', 'text'] + [f'{col}_processed' for col in text_columns] + ['combined_text']
    df[cols_to_save].to_csv(output_path, index=False, encoding='utf-8')
    print(f"   Saved {len(df)} records")
    
    # Финальная статистика
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample record:")
    print(f"  Original: {df[text_columns[0]].iloc[0][:100]}...")
    print(f"  Processed: {df['combined_text'].iloc[0][:100]}...")
    
    return df

if __name__ == '__main__':
    # Запускаем обработку
    process_russian_news_dataset(
        input_path='backend/data/raw/sample_news.csv',
        #input_path='backend/data/raw/russian_news1.csv',
        output_path='backend/data/processed/russian_news_processed.csv',
        text_columns=['title', 'text'],
        sample_size=None  # Обрабатываем все записи
    )