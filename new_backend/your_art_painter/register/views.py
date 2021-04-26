from django.http.response import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import UserForm
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

@csrf_exempt
def user_info(request):
    form = UserForm()
    # if request.method == "GET":
    #     print('success1')
    #     form = UserCreationForm(request.POST or None)
    #     if form.is_valid():
    #         print('success2')
    #         form.save()
    #         print(form)
    #     context = {'form': form}
    #     return render(request, 'register.html', context)
    # else:
    #     form = UserCreationForm(request.POST)
        
    if request.method == 'POST':
        print('It look OK--1')
        form = UserForm(request.POST or None)
        print('It look OK--2')
        print(form)
        print(form.errors)
        if form.is_valid():
            print('It look OK--3')
            form.save()
        #     print('It look OK--3')
        #     if User.objects.filter(email=form.cleaned_data['email']).exists():
        #         print('It look OK--4 exists email')
        #         return render(request, 'register.html', {
        #             'form': form,
        #             'error_message': 'Email already exists.'
        #         })
        #     elif form.cleaned_data['password'] != form.cleaned_data['confirm_password']:
        #         print('It look OK--5  password not match')
        #         return render(request, 'register.html', {
        #             'form': form,
        #             'error_,message': 'Password do not match.'
        #         })
        #     else:
        #         user = User.objects.create_user(
        #             form.cleaned_data['email'],
        #             form.cleaned_data['password']
        #         )
        #         user.save()

        #         return HttpResponseRedirect('homepage.html')
        # else:
        #     user_form = UserForm()
        context = {'form': form}

        return render(request, 'register.html', context)

    # if request.method == 'POST':
    #     Email = request.POST.get('email')
    #     Password = request.POST.get('password', 000000)
    #     Confirm_password = request.POST.get('confirm_password', 000000)
    #     print(Password, Confirm_password)
    #     if Password == Confirm_password:
    #         if User.objects.filter(Email=Email).exists():
    #             #messages.error(request, 'Email Already Exist')
    #             print('Email Already Exist')
    #         else:
    #             registerdata = register(Email=Email, Password=Password, Confirm_password=Confirm_password)
    #             registerdata.save()
    #             return redirect('homepage')
    #     else:
    #         #messages.error(request, 'Password and Confirm Password Not Matched')
    #         print("password not match")

    # return render(request, 'register.html', context = dict)
