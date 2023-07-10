from django.http import JsonResponse
from django.conf import settings
from django.shortcuts import render
import os
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import requests
import os
dataset_path = os.path.join(settings.MEDIA_ROOT, 'Images')

def home(request):
    filenames = []
    for i, fn in enumerate(sorted(os.listdir(dataset_path))):
        if i >= 50:  # only take the first 100 files
            break
        filename = os.path.join(settings.MEDIA_URL, 'Images', fn)
        print(filename)
        filenames.append(filename)
    context = {'filenames': filenames}
    return render(request, 'home.html', context)

def search_clip(request, shown=None, image_size=None):
    filenames = []
    print("Search_clip was called!")

    # if request.method == 'POST':
    #     query = request.POST.get('query', '') # Defaults to empty value if no query is detected to avoid exception.
    #     print("query value: ", query)

    # for i, fn in enumerate(sorted(os.listdir(dataset_path))):
    #     if i >= 5:  # only take the first 100 files
    #         break
    #     filename = os.path.join(settings.MEDIA_URL, 'Images', fn)
    #     filenames.append(filename)
    # context = {'filenames': filenames}
    # print(filenames)
    # return render(request, 'home.html', context)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print("CUDA Enabled.")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Prepare the text
    query = request.POST.get('query', '')
    print("query value: ", query)
    text = clip.tokenize([query]).to(device)

    # Prepare a list to store similarities
    similarities = []

    # Folder containing the images
    folder_path = os.path.join(settings.MEDIA_ROOT, 'Images/')

    # Counter for number of images processed
    counter = 0

    filenames = []
    for image_file in os.listdir(folder_path):
        # Full path to the image file
        image_path = os.path.join(folder_path, image_file)
        filenames.append(image_path)

        # Skip if not a file or not an image
        print(image_path)
        if not os.path.isfile(image_path) or not (image_file.endswith(".png") or image_file.endswith(".jpg")):
            continue

        # image_path = os.path.join(settings.MEDIA_URL, 'Images')
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            text_features = model.encode_text(text).cpu().numpy()
            image_features = model.encode_image(image).cpu().numpy()

        # Compute the similarity
        similarity = cosine_similarity(text_features, image_features)[0][0]
        similarities.append((image_file, similarity))

        print("Image: ", counter)
        counter += 1
        if counter >= 100:  # Stop processing after 100 images
            break

    # Sort the results by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Print the results
    for image_file, similarity in similarities:
        print(f"Image: {image_file}, Similarity: {similarity}")
        # TODO Sort by similarity and then show in UI, have to get Image path from image file.
        # TODO Overlay similarity over image.

    context = {'filenames': filenames}
    print(filenames)
    return render(request, 'home.html', context)

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
