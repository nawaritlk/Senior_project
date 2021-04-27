from django.db import models
from django.contrib.auth.models import User
from django.db.models.deletion import CASCADE
from django.db.models.fields import CharField
# from django.forms import widgets

# Create your models here.
class user_info(models.Model):
    #user = models.OneToOneField(User, on_delete=CASCADE)
    email = models.EmailField(blank=False)
    password = models.CharField(max_length=100, blank=False)
    confirm_password = models.CharField(max_length=100,blank=False)


    def __str__(self):
        return self.email