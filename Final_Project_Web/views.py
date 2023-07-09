from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

import requests
import os

def home(request):
    dataset_path = os.path.join(settings.MEDIA_ROOT, 'Images')

    filenames = []
    for i, fn in enumerate(sorted(os.listdir(dataset_path))):
        if i >= 1000:  # only take the first 100 files
            break
        filename = os.path.join(settings.MEDIA_URL, 'Images', fn)
        filenames.append(filename)
    context = {'filenames': filenames}
    return render(request, 'home.html', context)

def search_clip(request):
    # TODO: Implement your clip search functionality here
    return JsonResponse({})

def send_result(request, image_name):
    key_i = (image_name[-9:])[:5]
    my_obj = {'team': "name", 'item': key_i}
    # response = requests.get(url="https://siret.ms.mff.cuni.cz/lokoc/VBSEval/EndPoint.php", params=my_obj)
    # return JsonResponse({'result': response.text})
    # TODO Reactivate sending to endpoint.
    return 0

def find_similar(request, image_id, similar_images=None):
    # Implement functionality to find similar images here.
    # You'll need to return some kind of response, perhaps a list of similar image filenames.
    return JsonResponse({'similar_images': similar_images})
