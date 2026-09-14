from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('result/<int:prediction_id>/', views.result_view, name='result'),
    path('api/delete/<int:prediction_id>/', views.delete_prediction, name='delete_prediction'),
]
