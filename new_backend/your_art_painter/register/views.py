from django.shortcuts import render,redirect
from django.http import HttpResponse
from .models import register
from django.contrib.auth.models import User
from django.contrib import messages

# Create your views here.
def register(request):
    if request.method == 'POST':
        Email = request.POST.get('email')
        Password = request.POST.get('password', 000000)
        Confirm_password = request.POST.get('confirm_password', 000000)
        print(Password, Confirm_password)
        if Password == Confirm_password:
            if User.objects.filter(Email=Email).exists():
                #messages.error(request, 'Email Already Exist')
                print('Email Already Exist')
            else:
                registerdata = register(Email=Email, Password=Password, Confirm_password=Confirm_password)
                registerdata.save()
                return redirect('homepage')
        else:
            #messages.error(request, 'Password and Confirm Password Not Matched')
            print("password not match")


    return render(request, 'register.html')