# Generated by Django 3.0.5 on 2021-05-12 16:11

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('create_your_art', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='output',
            name='style',
            field=models.ForeignKey(default='1', on_delete=django.db.models.deletion.CASCADE, related_name='style', to='create_your_art.style'),
        ),
    ]
