from django.http.response import JsonResponse
from django.shortcuts import redirect, render, HttpResponseRedirect
from django.http import HttpResponse
from .models import upload, output, style
from django.contrib.auth.decorators import login_required
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse

from .utils import *
from .main import *



# model NST import
#!/usr/bin/env python3
import scipy.misc
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use ('TkAgg')
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, models
from torchvision.utils import save_image

import os
import cv2 as cv
import pandas as pd

import requests
from io import BytesIO,StringIO
import skimage.exposure as exposure
from skimage import io

# Create your views here.
@login_required
@csrf_exempt
def create(request):
    data = style.objects.all()
    context = {
        'data': data
    }
    

    return render(request, 'createYourArt.html', context)

@csrf_exempt
def submission(request):
    if request.method == 'POST':
        if request.is_ajax():
            style_id=request.POST.get('style_id')
            print("style",style_id)
        style_data = style.objects.get(pk=style_id)
        content_data = upload.objects.filter(user=request.user).order_by('-timestamp')[0]
        NST(request, style_data, content_data)
        
      
    return render(request,'submission.html')

@login_required
def file_upload_view(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            current_user = request.user
            my_file = request.FILES.get('file')
            
            imagedata = upload.objects.create(user=current_user,image=my_file)
            imagedata.save()

            return HttpResponseRedirect(reverse('homepage'))

    return JsonResponse({'post': 'false'})

def NST(request,style_data, content_data):
    style_url = style_data.image.url
    content_url = content_data.image.url
    current_user = request.user

    IMAGE_TYPE = 'url'
    STYLE_IMG = 'http://127.0.0.1:8000'+str(style_url)
    print(STYLE_IMG)
    # STYLE_IMG = 'http://127.0.0.1:8000/media/upload/'+str(style)
    CONTENT_IMG = 'http://127.0.0.1:8000'+str(content_url)
    ADAM_LR = 0.03 
    NUM_EPOCHS = 50
    # ADAM_LR = 0.003  
    # NUM_EPOCHS = 500
    STYLE_WEIGHT = 1e1
    CONTENT_WEIGHT = 1e-1
    IMG_SIZE = (224,224)
    CONTENT_WEIGHT = 1e-2
    STYLE_WEIGHT = 1e6
    MODEL_POOLING = 'max' # or 'avg'
    METHOD = 'after' # 'before'
    COLOR = None # 'histogram', 'luminance'

    generate = main(MODEL_POOLING,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,COLOR,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    generate_img = scipy.misc.toimage(generate)
    genIO = BytesIO()
    generate_img.save(genIO, format='JPEG')
    generate_image = InMemoryUploadedFile(genIO, None, '123.jpeg', 'media/upload',genIO.tell(), None)
    generateimg = output.objects.create(user=current_user,content=content_data,generate_img=generate_image,style=style_data)

    generateimg.save()