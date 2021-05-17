from django.db import models
from create_your_art.models import output
from django.contrib.auth.models import User

# Create your models here.
class Like(models.Model):
    post = models.ForeignKey(output, on_delete=models.CASCADE, related_name='liked_post')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='liker')
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '{} : {}'.format(self.user, self.post)

# class countingLike(models.Model):
#     total_like = models.IntegerField(default='0')
#     post = models.ForeignKey(output, on_delete=models.CASCADE, related_name='all_liked_post')
