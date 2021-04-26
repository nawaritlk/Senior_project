from django import forms
from django.contrib.auth.models import User
from register.models import user_info


class UserForm(forms.ModelForm):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput())
    confirm_password = forms.CharField(widget=forms.PasswordInput())

    class Meta():
        model = user_info
        fields = ('email', 'password', 'confirm_password')
