"""
Полный пайплайн обработки данных:
1. Загрузка сырых данных
2. Предобработка текста
3. Векторизация
4. Сохранение результатов

Запуск: python process_pipeline.py
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import json

# Добавление родительской директории в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.preprocessor import RussianTextPreprocessor, process_russian_news_dataset
from ml_engine.vectorizer import TextVectorizer, vectorize_dataset


class DataPipeline:
    """Класс для управления полным пайплайном обработки данных"""
    
    def __init__(self, config: dict):
        """
        Инициализация пайплайна
        
        Args:
            config: словарь с конфигурацией
        """
        self.config = config
        self.validate_config()
        
        # Создание необходимых директорий
        self.create_directories()
    
    def validate_config(self):
        """Проверка конфигурации"""
        required_keys = ['raw_data_path', 'processed_data_path', 'models_dir', 'text_columns']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if not os.path.exists(self.config['raw_data_path']):
            raise FileNotFoundError(f"Raw data file not found: {self.config['raw_data_path']}")
    
    def create_directories(self):
        """Создание необходимых директорий"""
        dirs = [
            os.path.dirname(self.config['processed_data_path']),
            self.config['models_dir'],
            'data/logs'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory ready: {dir_path}")
    
    def step1_preprocess(self):
        """Шаг 1: Предобработка текста"""
        print("\n" + "="*80)
        print("STEP 1: TEXT PREPROCESSING")
        print("="*80)
        
        df = process_russian_news_dataset(
            input_path=self.config['raw_data_path'],
            output_path=self.config['processed_data_path'],
            text_columns=self.config['text_columns'],
            sample_size=self.config.get('sample_size', None)
        )
        
        return df
    
    def step2_vectorize(self):
        """Шаг 2: Векторизация"""
        print("\n" + "="*80)
        print("STEP 2: TEXT VECTORIZATION")
        print("="*80)
        
        df, vectorizer = vectorize_dataset(
            preprocessed_csv_path=self.config['processed_data_path'],
            output_dir=self.config['models_dir'],
            text_column='combined_text',
            max_features=self.config.get('max_features', 5000)
        )
        
        return df, vectorizer
    
    def step3_save_metadata(self, df: pd.DataFrame):
        """Шаг 3: Сохранение дополнительных метаданных"""
        print("\n" + "="*80)
        print("STEP 3: SAVING ADDITIONAL METADATA")
        print("="*80)
        
        # Сохранение полного обработанного датасета (для анализа)
        full_output_path = self.config['processed_data_path'].replace('.csv', '_full.csv')
        print(f"\nSaving full processed dataset: {full_output_path}")
        df.to_csv(full_output_path, index=False, encoding='utf-8')
        
        # Сохранение JSON для API (компактная версия)
        api_columns = ['id', 'vector_index']
        for col in self.config['text_columns']:
            if col in df.columns:
                api_columns.append(col)
        
        # Добавление категории если есть
        if 'category' in df.columns:
            api_columns.append('category')
        
        api_metadata_path = os.path.join(self.config['models_dir'], 'api_metadata.json')
        print(f"Saving API metadata: {api_metadata_path}")
        df[api_columns].to_json(api_metadata_path, orient='records', indent=2, force_ascii=False)
        
        # Сохранение конфигурации пайплайна
        config_path = os.path.join(self.config['models_dir'], 'pipeline_config.json')
        print(f"Saving pipeline config: {config_path}")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        # Сохранение статистики датасета
        stats_path = os.path.join(self.config['models_dir'], 'dataset_stats.json')
        print(f"Saving dataset stats: {stats_path}")
        
        stats = {
            'total_records': int(len(df)),
            'columns': list(df.columns),
            'text_columns': self.config['text_columns'],
            'avg_text_length': float(df['combined_text'].str.len().mean()),
            'min_text_length': int(df['combined_text'].str.len().min()),
            'max_text_length': int(df['combined_text'].str.len().max())
        }
        
        # Статистика по категориям если есть
        if 'category' in df.columns:
            stats['categories'] = df['category'].value_counts().to_dict()
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print("\n✓ All metadata saved successfully!")
    
    def run(self):
        """Запуск полного пайплайна"""
        print("\n" + "="*80)
        print("STARTING FULL DATA PROCESSING PIPELINE")
        print("="*80)
        print(f"\nConfiguration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        try:
            # Шаг 1: Предобработка
            df = self.step1_preprocess()
            
            # Шаг 2: Векторизация
            df, vectorizer = self.step2_vectorize()
            
            # Шаг 3: Сохранение метаданных
            self.step3_save_metadata(df)
            
            # Финальный отчет
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY! ✓")
            print("="*80)
            print(f"\nProcessed {len(df)} records")
            print(f"Vocabulary size: {len(vectorizer.feature_names)}")
            print(f"\nOutput files:")
            print(f"  - Processed CSV: {self.config['processed_data_path']}")
            print(f"  - Models: {self.config['models_dir']}")
            print(f"  - Vectorizer: {os.path.join(self.config['models_dir'], 'tfidf_vectorizer.pkl')}")
            print(f"  - Vectors: {os.path.join(self.config['models_dir'], 'text_vectors.pkl')}")
            print(f"  - API Metadata: {os.path.join(self.config['models_dir'], 'api_metadata.json')}")
            
            return True
            
        except Exception as e:
            print("\n" + "="*80)
            print("PIPELINE FAILED! ✗")
            print("="*80)
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Process Russian News Dataset')
    parser.add_argument('--raw-data', type=str, default='backend/data/raw/lenta-ru-news.csv',
                       help='Path to raw CSV file')
    parser.add_argument('--processed-data', type=str, default='backend/data/processed/russian_news_processed.csv',
                       help='Path to save processed CSV')
    parser.add_argument('--models-dir', type=str, default='backend/ml_engine/models',
                       help='Directory to save models')
    parser.add_argument('--title-column', type=str, default='title',
                       help='Name of title column')
    parser.add_argument('--text-column', type=str, default='text',
                       help='Name of text column')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Sample size for testing (None = all data)')
    parser.add_argument('--max-features', type=int, default=5000,
                       help='Maximum number of features for TF-IDF')
    
    args = parser.parse_args()
    
    # Конфигурация
    config = {
        'raw_data_path': args.raw_data,
        'processed_data_path': args.processed_data,
        'models_dir': args.models_dir,
        'text_columns': [args.title_column, args.text_column],
        'sample_size': args.sample_size,
        'max_features': args.max_features
    }
    
    # Запуск пайплайна
    pipeline = DataPipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
