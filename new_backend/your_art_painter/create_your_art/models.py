from django.db import models
from django.contrib.auth.models import User

from django.core.files import File
from urllib import request
import os
import uuid

# Create your models here.
# class upload(models.Model):

#     timestamp = models.DateTimeField(auto_now_add=True)
#     user = models.ForeignKey(User, on_delete=models.CASCADE)
#     image = models.ImageField(upload_to='upload/')
#     output = models.ImageField(upload_to='output')

#     # def __str__(self):
#     #     return self.user
def get_file_upload(instance, filename):
    ext = filename.split('.')[-1]
    filename = "%s.%s" % (uuid.uuid4(), ext)
    return os.path.join(instance.directory_string_var, filename)

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
    image = models.ImageField(upload_to = get_file_upload)
    directory_string_var = 'upload'

class output(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete = models.CASCADE)
    generate_img = models.ImageField(upload_to = get_file_output)
    directory_string_var = 'output'

class generateNST(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete = models.CASCADE)
    content = models.ForeignKey(upload, on_delete= models.CASCADE)
    generate = models.ForeignKey(output, on_delete= models.CASCADE)

class style(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    style = models.ImageField(upload_to = get_file_style)
    directory_string_var = 'style'