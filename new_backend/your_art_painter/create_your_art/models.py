from django.db import models
from django.contrib.auth.models import User


from django.core.files import File
from urllib import request
import os
import uuid

# Create your models here.


def get_file_output(instance, filename):
    ext = filename.split('.')[-1]
    filename = "%s.%s" % (uuid.uuid4(), ext)
    return os.path.join(instance.directory_string_var, filename)
    

def get_file_style(instance, filename):
    ext = filename.split('.')[-1]
    filename = "%s.%s" % (uuid.uuid4(), ext)
    return os.path.join(instance.directory_string_var, filename)

class upload(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete = models.CASCADE)
    image = models.ImageField(upload_to = 'upload/')
    

class output(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete = models.CASCADE, related_name='user')
    generate_img = models.ImageField(upload_to = get_file_output)
    content = models.ForeignKey('create_your_art.upload',on_delete= models.CASCADE, related_name='content')
    style = models.ForeignKey('create_your_art.style',on_delete= models.CASCADE, related_name='style', default='1')
    total_like = models.IntegerField(default='0')
    public = models.BooleanField(default=True)
    directory_string_var = 'output'


class style(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to = get_file_style)
    directory_string_var = 'style'