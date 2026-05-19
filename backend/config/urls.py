"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from recs_api import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/upload/', views.upload_dataset, name='upload_dataset'),
    path('api/search/', views.search_recommendations, name='search_recommendations'),
    path('api/similar/', views.similar_recommendations, name='similar_recommendations'),
    path('api/history-recs/', views.history_recommendations, name='history_recommendations'),
    path('api/auth/register/', views.auth_register, name='auth_register'),
    path('api/auth/login/', views.auth_login, name='auth_login'),
    path('api/auth/logout/', views.auth_logout, name='auth_logout'),
    path('api/auth/me/', views.auth_me, name='auth_me'),
    path('api/auth/history/', views.auth_history, name='auth_history'),
    path('api/auth/history/clear/', views.auth_history_clear, name='auth_history_clear'),
    path('api/auth/history/delete/', views.auth_history_delete, name='auth_history_delete'),
]
