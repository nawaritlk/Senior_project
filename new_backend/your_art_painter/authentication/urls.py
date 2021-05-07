from django.contrib import admin
from django.urls import path,include
from . import views
from django.contrib.auth import views as auth_views
from django.conf.urls import *

admin.autodiscover()

urlpatterns = [
    path("register/", views.register),
    # path("",views.homepage, name= 'homepage'),
    path("authlogin/", views.authlogin, name = 'authlogin'),
    path("logout/", views.authlogout, name = 'authlogout'),
    path("forgetpassword/", views.forgetpassword, name = 'forgetpassword'),
    path("login/", views.login_before, name = 'login'),

    #reset password
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name="forgetpassword/password_reset_form.html"), name = 'password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name="forgetpassword/password_reset_done.html"), name = 'password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name="forgetpassword/password_reset_confirm.html"), name = 'password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name="forgetpassword/password_reset_complete.html"), name = 'password_reset_complete'),
]
