from word_processing.views import word_page
from django.urls import path

urlpatterns = [
    path('', word_page, name='word'),
]