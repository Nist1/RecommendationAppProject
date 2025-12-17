import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from ml_engine.process_pipeline import DataPipeline
from ml_engine.recomendation import ContentRecommender


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'russian_news_processed.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'ml_engine', 'models')


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
        pipeline.run()
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

    try:
        recommender = ContentRecommender(models_dir=MODELS_DIR)
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