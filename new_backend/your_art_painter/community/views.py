from your_art_painter.create_your_art.models import output
from django.shortcuts import render
from django.http import HttpResponse
from your_art_painter.create_your_painter.models import output

# from django.apps import apps
# all_model = apps.get_models('create_your_art', 'output')


# Create your views here.
def community(request):
    # data = output.objects.all()
    # print(all_model.output.content)
    # output2 = output1.objects.select_related().all()
    # for i in output2:
    #   print(i.content)
    context={
      # 'data': data
    
    }
    return render(request, 'community.html', context)