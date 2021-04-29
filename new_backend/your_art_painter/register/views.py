from django.http.response import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
# from .forms import UserForm
from django.views.decorators.csrf import csrf_exempt

# Create your views here.


@csrf_exempt
def user_info(request):
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
                # return redirect('homepage')

        else:
            messages.error(request, 'Password and Confirm Password Not Matched')
            # print("password not match")

    return render(request, 'register.html')
