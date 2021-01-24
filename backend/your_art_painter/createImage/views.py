from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def create(request):
    return render(request, 'createYourArt.html')

def submission(request):
    return render(request, 'submission.html')