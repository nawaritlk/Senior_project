from django.contrib import admin
from .models import upload,output,generateNST,style

# Register your models here.
admin.site.register(upload)
admin.site.register(output)
admin.site.register(generateNST)
admin.site.register(style)