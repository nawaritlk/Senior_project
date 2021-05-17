from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path("",views.community, name='community'),
    path("liked/<pk>/", views.liked, name='liked'),
    path("unliked/<pk>/", views.unliked, name='unliked'),

]