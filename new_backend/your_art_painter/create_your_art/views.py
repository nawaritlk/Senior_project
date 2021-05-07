from django.http.response import JsonResponse
from django.shortcuts import render
from django.http import HttpResponse
from .models import upload
from django.contrib.auth.decorators import login_required


# Create your views here.
@login_required
def create(request):
    return render(request, 'createYourArt.html')

def submission(request):
    return render(request, 'submission.html')

def file_upload_view(request):
    # print(request.FILES)
    if request.user.is_authenticated:
        if request.method == 'POST':
            current_user = request.user
            my_file = request.FILES.get('file')
            print(my_file)
            imagedata = upload.objects.create(user=current_user,image=my_file)
            imagedata.save()
            return HttpResponse('')
    return JsonResponse({'post': 'false'})



