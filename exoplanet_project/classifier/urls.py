from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('train/', views.train_new_model, name='train'),
    path('predict/', views.predict_view, name='predict'),
    path('predictions/', views.predictions_list, name='predictions_list'),
    path('comparison/', views.model_comparison, name='comparison'),
]