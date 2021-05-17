from django.shortcuts import render,HttpResponseRedirect
from django.http import HttpResponse
from create_your_art.models import upload,output,style
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from .models import Like

# from django.apps import apps
# all_model = apps.get_models('create_your_art', 'output')


# Create your views here.
def community(request):
    data = output.objects.all()
    liked_post = Like.objects.filter(user=request.user)
    liked_post_list = liked_post.values_list('post', flat=True)
    # print(data)
    # for i in output2:
    #   print(i.content)
    context={
      'data': data,
      'like_post_list': liked_post_list
    
    }
    return render(request, 'community.html', context)

@login_required
def liked(request, pk):
    post = output.objects.get(pk=pk)
    already_liked = Like.objects.filter(post=post, user=request.user)
    if not already_liked:
        liked_post = Like(post=post, user=request.user)
        liked_post.save()
    return HttpResponseRedirect(reverse('community'))

@login_required
def unliked(request, pk):
    post = output.objects.get(pk=pk)
    already_liked = Like.objects.filter(post=post,user=request.user)
    already_liked.delete()
    return HttpResponseRedirect(reverse('community'))
