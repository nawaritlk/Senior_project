from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def profile(request):
    return render(request, 'authentication/profile.html')