import os
import json
import threading
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate
from django.contrib.auth.models import User

from ml_engine.process_pipeline import DataPipeline
from ml_engine.recomendation import ContentRecommender
from ml_engine.embedding_recommender import EmbeddingRecommender, build_sbert_vectors
from recs_api.models import SearchHistory, AuthToken


def get_user_from_request(request):
    """Получить пользователя по токену из заголовка Authorization."""
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Token '):
        return None
    key = auth[6:]
    try:
        return AuthToken.objects.select_related('user').get(key=key).user
    except AuthToken.DoesNotExist:
        return None


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


# --- Auth ---

@csrf_exempt
def auth_register(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)

    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)

    username = body.get('username', '').strip()
    password = body.get('password', '').strip()

    if not username or not password:
        return JsonResponse({'status': 'error', 'message': 'Username and password required'}, status=400)

    if User.objects.filter(username=username).exists():
        return JsonResponse({'status': 'error', 'message': 'Пользователь уже существует'}, status=400)

    user = User.objects.create_user(username=username, password=password)
    token = AuthToken.generate(user)
    return JsonResponse({'status': 'success', 'username': user.username, 'token': token})


@csrf_exempt
def auth_login(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)

    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)

    username = body.get('username', '').strip()
    password = body.get('password', '').strip()

    user = authenticate(request, username=username, password=password)
    if user is None:
        return JsonResponse({'status': 'error', 'message': 'Неверный логин или пароль'}, status=401)

    token = AuthToken.generate(user)
    return JsonResponse({'status': 'success', 'username': user.username, 'token': token})


@csrf_exempt
def auth_logout(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)
    user = get_user_from_request(request)
    if user:
        AuthToken.objects.filter(user=user).delete()
    return JsonResponse({'status': 'success'})


@csrf_exempt
def auth_me(request):
    user = get_user_from_request(request)
    if user:
        return JsonResponse({'status': 'success', 'username': user.username})
    return JsonResponse({'status': 'anonymous'})


@csrf_exempt
def auth_history_delete(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)
    user = get_user_from_request(request)
    if not user:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)

    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)

    query = body.get('query', '').strip()
    SearchHistory.objects.filter(user=user, query=query).delete()
    return JsonResponse({'status': 'success'})


@csrf_exempt
def auth_history_clear(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)
    user = get_user_from_request(request)
    if not user:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)

    SearchHistory.objects.filter(user=user).delete()
    return JsonResponse({'status': 'success'})


@csrf_exempt
def auth_history(request):
    if request.method != 'GET':
        return JsonResponse({'status': 'error', 'message': 'Only GET allowed'}, status=405)
    user = get_user_from_request(request)
    if not user:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)

    entries = SearchHistory.objects.filter(user=user).values('query', 'timestamp')[:20]
    total = SearchHistory.objects.filter(user=user).count()

    history = [
        {
            'query': e['query'],
            'timestamp': e['timestamp'].strftime('%d.%m.%Y %H:%M'),
        }
        for e in entries
    ]

    return JsonResponse({'status': 'success', 'history': history, 'total': total})


# --- Dataset ---

@csrf_exempt
def upload_dataset(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)

    file = request.FILES.get('dataset')
    if not file:
        return JsonResponse({'status': 'error', 'message': 'No file provided'}, status=400)

    os.makedirs(RAW_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, 'uploaded_dataset.csv')

    with open(raw_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

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


# --- Recommendations ---

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

    user = get_user_from_request(request)
    if user:
        SearchHistory.objects.create(user=user, query=query)

    try:
        recommender = _get_recommender(method)
        df = recommender.get_recommendations_for_user(query, n=5)
        return JsonResponse({'status': 'success', 'results': serialize_recommendations(df)})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def similar_recommendations(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)

    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)

    item_id = body.get('item_id')
    n = body.get('n', 3)

    try:
        item_id = int(item_id)
        n = int(n)
    except (TypeError, ValueError):
        return JsonResponse({'status': 'error', 'message': 'Invalid item_id or n'}, status=400)

    if n < 1:
        return JsonResponse({'status': 'error', 'message': 'n must be positive'}, status=400)

    try:
        recommender = ContentRecommender(models_dir=MODELS_DIR)
        df = recommender.get_similar_items(item_id, n=min(n, 5))
        return JsonResponse({'status': 'success', 'results': serialize_recommendations(df)})
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

    method = body.get('method', 'tfidf')
    exclude_ids = set(body.get('exclude_ids', []))

    user = get_user_from_request(request)
    if user:
        history = list(
            SearchHistory.objects.filter(user=user)
            .values_list('query', flat=True)[:3]
        )
    else:
        history = body.get('history', [])

    if not history:
        return JsonResponse({'status': 'success', 'results': []})

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
                    combined.append(serialize_recommendation_row(row))

        return JsonResponse({'status': 'success', 'results': combined})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
