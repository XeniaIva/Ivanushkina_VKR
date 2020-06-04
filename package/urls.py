from package.views import package
from django.urls import path

urlpatterns = [
    path('', package, name='package'),
]