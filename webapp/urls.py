from django.urls import path
from webapp import views

urlpatterns = [
    path('', views.index),
    path('login', views.login),
    path('logout', views.index),
    path('retrain', views.retrain),
    path('start', views.start),
]