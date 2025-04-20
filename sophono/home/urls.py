from django.urls import path
from . import views

urlpatterns = [
    # Home page
    path('', views.home_view, name='home'),
    
    # Processing endpoints
    path('process-music/', views.process_music, name='process_music'),
    path('process-lyrics/', views.process_lyrics, name='process_lyrics'),
]