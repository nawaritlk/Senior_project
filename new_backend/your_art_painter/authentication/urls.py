from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path("register/", views.register),
    # path("",views.homepage, name= 'homepage'),
    path("authlogin/", views.authlogin, name = 'authlogin'),
    path("logout/", views.authlogout, name = 'authlogout')
]
