from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('homepage.urls')),
    path('create/', include('create_your_art.urls')),
    path('submission/', include('create_your_art.urls')),
    path('profile/', include('user_info.urls')),
    path('community/', include('community.urls')),
    path('register/', include('register.urls')),
    path('login/', include('authentication.urls')),
]
