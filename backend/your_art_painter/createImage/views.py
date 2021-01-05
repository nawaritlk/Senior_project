from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def create(request):
    return HttpResponse("This is create page")

def submission(request):
    return HttpResponse("This is submission page")