from django.db import models

# Create your models here.
class register(models.Model):
    email = models.CharField(max_length=100, blank=False)
    password = models.CharField(max_length=100, blank=False)
    confirm_password = models.CharField(max_length=100, blank=False)
    def __str__(self):
        return self.email
