from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class upload(models.Model):

    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='upload/')
    # output = models.ImageField(upload_to='output')

    # def __str__(self):
    #     return self.user

class output(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    generate_img = models.ImageField(upload_to='output/')