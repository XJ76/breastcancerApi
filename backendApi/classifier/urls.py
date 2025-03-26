from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.PredictBreastCancerView.as_view(), name='predict'),
    path('metrics/', views.ModelMetricsView.as_view(), name='metrics'),
]

