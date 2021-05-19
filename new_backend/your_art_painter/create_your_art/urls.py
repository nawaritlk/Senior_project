from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path("",views.create ),
    path("submission/", views.submission, name = "submission"),
    path("upload/", views.file_upload_view, name="upload-view"),
    
    
]
