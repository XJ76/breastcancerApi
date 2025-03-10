from django.urls import path
from .views import PredictBreastCancerView

urlpatterns = [
    path('predict/', PredictBreastCancerView.as_view(), name='predict'),
]

