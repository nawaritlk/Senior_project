from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path("",views.temp),
    path("upload_temp/",views.file_upload_view_temp, name="upload-temp")
]