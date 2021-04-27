from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.contrib.auth import authenticate, login

# Create your views here.
def login(request):
    if request.method == 'POST':
        email = request.POST.get['email']
        password = request.POST.get['password']
        user = authenticate(request, email=email, password=password)
        if user is not None:
            login(request,user)
            return redirect('register')
        else:
            print('invalid user')

    return render(request, 'login.html')
    
def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password', 000000)
        confirm_password = request.POST.get('confirm_password', 000000)
        if password == confirm_password:
            if User.objects.filter(email=email).exists():
                messages.error(request, 'Email Already Exist')
                # print('Email Already Exist')
            else:
                registerdata = User.objects.create_user(username=username, email=email, password=password)
                registerdata.save()
                login(request, User)
                # return redirect('homepage')
        else:
            messages.error(request, 'Password and Confirm Password Not Matched')
            # print("password not match")
    return render(request, 'register.html')

