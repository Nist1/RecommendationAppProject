import os
import json
import threading
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from ml_engine.process_pipeline import DataPipeline
from ml_engine.recomendation import ContentRecommender
from ml_engine.embedding_recommender import EmbeddingRecommender, build_sbert_vectors


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'russian_news_processed.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'ml_engine', 'models')


_recommenders = {}
_lock = threading.Lock()

def _get_recommender(method: str):
    if method not in _recommenders:
        with _lock:
            if method not in _recommenders:
                if method == 'sbert':
                    _recommenders[method] = EmbeddingRecommender(models_dir=MODELS_DIR)
                else:
                    _recommenders[method] = ContentRecommender(models_dir=MODELS_DIR)
    return _recommenders[method]


def _reset_recommenders():
    """Сбрасывает кэш после загрузки нового датасета."""
    _recommenders.clear()


def serialize_recommendations(df):
    return [serialize_recommendation_row(row) for _, row in df.iterrows()]


def serialize_recommendation_row(row):
    return {
        'id': int(row['vector_index']),
        'title': row.get('title', ''),
        'text': row.get('text', ''),
        'similarity': float(row['similarity']),
    }


@csrf_exempt
def upload_dataset(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)

    file = request.FILES.get('dataset')
    if not file:
        return JsonResponse({'status': 'error', 'message': 'No file provided'}, status=400)

    os.makedirs(RAW_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, 'uploaded_dataset.csv')

    # Сохранить загруженный файл
    with open(raw_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    # Запустить пайплайн (предобработка + векторизация)
    try:
        config = {
            'raw_data_path': raw_path,
            'processed_data_path': PROCESSED_PATH,
            'models_dir': MODELS_DIR,
            'text_columns': ['title', 'text'],
            'sample_size': 1000,
            'max_features': 5000,
        }
        pipeline = DataPipeline(config)
        success = pipeline.run()
        if not success:
            return JsonResponse({'status': 'error', 'message': 'Pipeline failed. Check server logs.'}, status=500)

        build_sbert_vectors(MODELS_DIR)
        _reset_recommenders()
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'success'})

@csrf_exempt
def search_recommendations(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)

    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)

    query = body.get('query', '').strip()
    if not query:
        return JsonResponse({'status': 'error', 'message': 'Query is empty'}, status=400)

    method = body.get('method', 'tfidf')

    try:
        recommender = _get_recommender(method)
        df = recommender.get_recommendations_for_user(query, n=5)

        results = []
        for _, row in df.iterrows():
            results.append({
                'id': int(row['vector_index']),
                'title': row.get('title', ''),
                'text': row.get('text', ''),
                'similarity': float(row['similarity']),
            })

        return JsonResponse({'status': 'success', 'results': results})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
def history_recommendations(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)

    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)

    history = body.get('history', [])
    exclude_ids = set(body.get('exclude_ids', []))
    method = body.get('method', 'tfidf')

    if not history:
        return JsonResponse({'status': 'success', 'results': []})

    history = history[-3:]

    try:
        recommender = _get_recommender(method)
        seen_ids = set(exclude_ids)
        combined = []

        for query in history:
            query = query.strip()
            if not query:
                continue
            df = recommender.get_recommendations_for_user(query, n=3)
            for _, row in df.iterrows():
                doc_id = int(row['vector_index'])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    combined.append({
                        'id': doc_id,
                        'title': row.get('title', ''),
                        'text': row.get('text', ''),
                        'similarity': float(row['similarity']),
                    })

        return JsonResponse({'status': 'success', 'results': combined})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)