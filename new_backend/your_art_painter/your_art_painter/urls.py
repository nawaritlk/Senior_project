from django.contrib import admin
from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('homepage.urls')),
    path('create/', include('create_your_art.urls')),
    path('submission/', include('create_your_art.urls')),
    path('profile/', include('user_info.urls')),
    path('community/', include('community.urls')),
    # path('register/', include('authentication.urls')),
    path('auth/', include('authentication.urls')),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns += staticfiles_urlpatterns()
