# Generated by Django 3.0.5 on 2021-04-25 10:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('register', '0002_auto_20210425_1724'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userinfo',
            name='user',
        ),
    ]
