
from django.shortcuts import render, redirect, HttpResponseRedirect
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from create_your_art.models import upload,output,style
from django.contrib.auth.decorators import login_required
from django.urls import reverse


@csrf_exempt
# Create your views here.
def authlogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username, password)
        user = authenticate(request, username=username, password=password)
        print(user)
        if user is not None:
            login(request,user)
            return redirect('homepage')
            print('login successful')
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
                # login(request, User)
                return redirect('authlogin')
        else:
            messages.error(request, 'Password and Confirm Password Not Matched')
            # print("password not match")
    return render(request, 'register.html')

def authlogout(request):
    logout(request)
    return redirect('authlogin')

def homepage(request):
    return render(request, 'authentication/homepage.html')

def forgetpassword(request):
    return render(request, 'forgetPassword.html')


@csrf_exempt
def login_before(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username, password)
        user = authenticate(request, username=username, password=password)
        print(user)
        if user is not None:
            login(request,user)
            return redirect('homepage')
        else:
            print('invalid user')
    return render(request, 'login_before.html')

@login_required
def profile(request):
    data = output.objects.filter(user=request.user)


    print("data:", data)
    context={
      'data': data,
    #   'like_post_list': liked_post_list
    
    }
    return render(request, 'profile.html', context)

@login_required   
def delete_output(request, pk):
    generated = output.objects.get(pk=pk)
    print(generated)
    generated.delete()
    return HttpResponseRedirect(reverse('profile'))

@login_required
def make_private(request, pk):
    generated = output.objects.get(pk=pk)
    generated.public=False
    generated.save()
    return HttpResponseRedirect(reverse('profile'))

@login_required
def make_public(request, pk):
    generated = output.objects.get(pk=pk)
    generated.public=True
    generated.save()
    return HttpResponseRedirect(reverse('profile'))
