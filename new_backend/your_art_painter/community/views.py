from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def community(request):
    return render(request, 'community.html')