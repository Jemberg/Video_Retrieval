from django.http import JsonResponse
from django.conf import settings
from django.shortcuts import render
from PIL import Image
from open_clip import tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

import torch
import clip
import cv2
import numpy as np
import requests
import os
import math
import open_clip

import os
import torch
from PIL import Image
import open_clip

import matplotlib.pyplot as plt
from sklearn_som.som import SOM
import numpy as np

dataset_path = os.path.join(settings.MEDIA_ROOT, 'Images')
folder_path = os.path.join(settings.MEDIA_ROOT, 'Images/')

import os
import torch
import numpy as np
from sklearn_som.som import SOM
from django.shortcuts import render
from django.conf import settings

from django.shortcuts import render
from django.conf import settings
from django.core.paginator import Paginator
import os
import torch
import numpy as np
from sklearn_som.som import SOM

import pickle
import os

imagesPerPage = 1000

# Load the model and tokenizer when the module is loaded
print("Loading model and tokenizer.")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("Loading model and tokenizer done.")

def home(request):
    # Directory of features
    features_dir = 'Features'
    clusters_file = 'clusters.pkl'

    # Gather all the feature vectors into a list
    all_features = []
    filenames = []
    counter = 0

    # Check if cluster assignments have already been calculated
    if os.path.exists(clusters_file):
        with open(clusters_file, 'rb') as f:
            filename_cluster_zip = pickle.load(f)
    else:
        # Loop over all feature files in the directory
        for feature_file in sorted(os.listdir(features_dir)):
            # Load the features from disk
            features = torch.load(os.path.join(features_dir, feature_file))

            # Append the features to the list
            all_features.append(features.numpy())

            # Append the corresponding image filename to the filenames list
            image_file = feature_file.replace('.pt', '')
            filenames.append(os.path.join(settings.MEDIA_URL, 'Images', image_file))
            counter += 1

        # Stack all the feature vectors into a 2D numpy array
        data = np.vstack(all_features)

        # Determine the number of features from the shape of the data array
        n_features = data.shape[1]

        # Initialize a SOM with the correct number of features
        som = SOM(m=25, n=40, dim=n_features)

        # Fit the SOM to the data and get the cluster assignments
        cluster_assignments = som.fit_predict(data)

        # Create a list of (filename, cluster_assignment) tuples
        filename_cluster_zip = list(zip(filenames, cluster_assignments))

        # Sort the list based on cluster assignments
        filename_cluster_zip.sort(key=lambda x: x[1])

        # Save the cluster assignments for future use
        with open(clusters_file, 'wb') as f:
            pickle.dump(filename_cluster_zip, f)

    # Pagination
    paginator = Paginator(filename_cluster_zip, imagesPerPage)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    page_amount = paginator.num_pages

    context = {'filenames': page_obj, 'total_pages': page_amount}
    return render(request, 'home.html', context)

# def search_clip(request, shown=None, image_size=None):
#     filenames = []
#
#     # Load the model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if torch.cuda.is_available(): print("CUDA Enabled.")
#     model, preprocess = clip.load("ViT-B/32", device=device)
#
#     # Prepare the text
#     query = request.POST.get('query', '')
#     print("query value: ", query)
#     text = clip.tokenize([query]).to(device)
#
#     # Prepare a list to store similarities as well as images.
#     similarities = []
#     similarityExcl = []
#     # Exclusively only stores the numbers for the similarities
#
#     # # Folder containing the images
#     # folder_path = os.path.join(settings.MEDIA_ROOT, 'Images/')
#
#     # Counter for number of images processed
#     counter = 0
#
#     filenames = []
#     for image_file in os.listdir(folder_path):
#         # Full path to the image file
#         image_path = os.path.join(folder_path, image_file)
#
#         # Skip if not a file or not an image
#         print(image_path)
#         if not os.path.isfile(image_path) or not (image_file.endswith(".png") or image_file.endswith(".jpg")):
#             continue
#
#         # image_path = os.path.join(settings.MEDIA_URL, 'Images')
#         image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#
#         # Calculate features
#         with torch.no_grad():
#             text_features = model.encode_text(text).cpu().numpy()
#             image_features = model.encode_image(image).cpu().numpy()
#
#         # Compute the similarity
#         similarity = cosine_similarity(text_features, image_features)[0][0]
#         similarities.append((image_path, similarity))
#
#     # Sort the results by similarity in descending order
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     print(similarities)
#
#     # Print the results
#     for image_path, similarity in similarities:
#         print(f"Image: {image_path}, Similarity: {similarity}")
#         filenames.append(image_path)
#         similarityExcl.append("{:.3f}".format(similarity * 100) + "%")
#
#     filename_similarity_zip = zip(filenames, similarityExcl)
#     context = {'filenames': filename_similarity_zip, "query": query}
#     return render(request, 'home.html', context)

def send_result(request):
    # TODO Fix send_result not being found
    print("Sent response to server.")
    # key_i = (image_name[-9:])[:5]
    my_obj = {'team': "Martin", 'item': "21354"}
    # Query that worked: https://siret.ms.mff.cuni.cz/lokoc/VBSEval/EndPoint.php?team=Martin&item=24563
    # Query to check on: https://siret.ms.mff.cuni.cz/lokoc/VBSEval/eval.php
    response = requests.get(url="https://siret.ms.mff.cuni.cz/lokoc/VBSEval/EndPoint.php", params=my_obj, verify=False)
    print(response)
    return JsonResponse({'result': response.text})

# TODO find_similar has to be implemented properly.
def find_similar(request, image_id, similar_images=None):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)

    # Load the query image
    query_image = preprocess(Image.open("path_to_your_query_image.jpg")).unsqueeze(0).to(device)

    # Calculate the features of the query image
    with torch.no_grad():
        query_features = model.encode_image(query_image)

    # Prepare a list to store the filenames and similarities
    image_similarities = []

    # Go through all images in the directory
    for filename in os.listdir('path_to_your_image_directory'):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load and preprocess the image
            image = preprocess(Image.open(os.path.join('path_to_your_image_directory', filename))).unsqueeze(0).to(device)

            # Calculate the features of the image
            with torch.no_grad():
                image_features = model.encode_image(image)

            # Calculate the similarity between the query image and the image
            similarity = (query_features @ image_features.T).item()

            # Add the filename and similarity to the list
            image_similarities.append((filename, similarity))

    # Sort the images by similarity
    image_similarities.sort(key=lambda x: x[1], reverse=True)

    # Print the images sorted by similarity
    for filename, similarity in image_similarities:
        print(f'{filename}: {similarity}')

    return JsonResponse({'similar_images': similar_images})

# TODO find_similar_histogram needs to be properly implemented, requires button in front-end as well.
def find_similar_histogram(request):
    # Specify the directory of your images and the path to your query image
    directory = "path_to_your_image_directory"
    query_image_path = "path_to_your_query_image.jpg"

    # Specify the number of bins per channel for the histogram
    bins = [8, 8, 8]

    # Load the query image and compute its histogram
    query_image = cv2.imread(query_image_path)
    query_hist = cv2.calcHist([query_image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

    # Normalize the histogram
    query_hist = cv2.normalize(query_hist, query_hist).flatten()

    # Prepare a list to store the filenames and similarities
    image_similarities = []

    # Go through all images in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image
            image = cv2.imread(os.path.join(directory, filename))

            # Compute the histogram of the image
            hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

            # Normalize the histogram
            hist = cv2.normalize(hist, hist).flatten()

            # Compute the cosine similarity between the query image's histogram
            # and this image's histogram
            similarity = 1 - distance.cosine(query_hist, hist)

            # Add the filename and similarity to the list
            image_similarities.append((filename, similarity))

    # Sort the images by similarity
    image_similarities.sort(key=lambda x: x[1], reverse=True)

    # Print the top 5 images sorted by similarity
    for filename, similarity in image_similarities[:5]:
        print(f'{filename}: {similarity}')

def combined_clip(request):
    print("Combined clip function called")
    print()
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Prepare the text
    text = clip.tokenize(["A dog on a skateboard"]).to(device)

    # Prepare the image
    image = preprocess(Image.open("dog_on_skateboard.jpg")).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        text_features = model.encode_text(text)
        image_features = model.encode_image(image)

    # Combine the features
    combined_features = (text_features + image_features) / 2  # Here we average the features, but you can combine them in other ways

    # Compare to database
    # This depends on how your database is structured, but you would compare `combined_features` to the features of each image in the database
    # For example, if `database_features` is a 2D tensor where each row is the features of one image:
    similarities = torch.nn.functional.cosine_similarity(combined_features, database_features)

    # Sort by similarity
    ranked_indices = torch.argsort(similarities, descending=True)

    # Now `ranked_indices[0]` is the index of the most similar image, `ranked_indices[1]` is the second most similar, and so on

# def L2S(x1, y1, x2, y2):
#     return (x1 - x2) ** 2 + (y1 - y2) ** 2
#
# def UpdateScores(X, Y, scores, display, likeID, alpha):
#     for i in range(0, X.size):
#         PF = math.exp(-L2S(X[likeID], Y[likeID], X[i], Y[i]) / alpha)
#         NF = 0
#         for j in display:
#             if j != likeID:
#                 NF += math.exp(-L2S(X[j], Y[j], X[i], Y[i]) / alpha)
#         scores[i] = scores[i] * PF / NF
#
# def DrawDataANdScores(X, Y, scores, display, likeID):
#     colors = []
#     for i in range(0, X.size):
#         c = int(scores[i] * 255)
#         if i == likeID: colors.append("red")
#         elif i in display: colors.append("blue")
#         else: colors.append('#%02x%02x%02x' % (0, 255 - c, 0))

# TODO Implement bayesian feedback loop.
# def feedback_loop(request):
#     print("Feedback update called.")
#     positive_image = request.POST.get('lastSelected', "")
#     print("Positive example: ", positive_image)
#
#     # TODO Precompute all the distances with matrix multiplication, is 100x faster than iterating.
#     # TODO Change positive_image to correct solution later.
#
#     # TODO Negative examples can be 20-30 images, does not have to be all of the rest of the images.
#     context = {'filenames': positive_image}
#     return render(request, 'home.html', context)

from django.shortcuts import render
import torch
import numpy as np
import os
import math

def L2S(feature1, feature2):
    return np.sum((feature1 - feature2) ** 2)

from annoy import AnnoyIndex

# Speed up the process by using an approximate nearest neighbors search algorithm instead of computing all pairwise distances.
# Constructs an index with Annoy and then queries it to get the distance between vectors
def UpdateScores(features, scores, display, likeID, alpha):
    # Convert lists to numpy arrays for vectorized operations
    features = np.array(features)
    scores = np.array(scores)

    # Compute the squared L2 norm between the liked feature and all other features
    diff = features - features[likeID]
    L2_norms = np.sum(diff ** 2, axis=1)

    # Compute the positive factor (PF)
    PF = np.exp(-L2_norms / alpha)

    # Compute the negative factor (NF)
    NF = np.zeros_like(scores)
    for j in display:
        if j != likeID:
            diff = features - features[j]
            L2_norms = np.sum(diff ** 2, axis=1)
            NF += np.exp(-L2_norms / alpha)

    # Update the scores
    scores = scores * PF / (NF + 1e-9)  # add a small constant to avoid division by zero
    return scores



def feedback_loop(request):
    # Directory of features
    image_dir = 'Images'
    features_dir = 'Features'
    # Fetch the ID of the liked image from the request
    likeID = int(request.POST.get('likeID')[:-4])
    print(likeID)

    # Load all the feature vectors into memory
    features = []
    print("Loading features")
    for image_file in sorted(os.listdir(image_dir)):
        features_path = os.path.join(features_dir, image_file + '.pt')
        if os.path.exists(features_path):
            # Load the feature vector and add it to the list
            image_features = torch.load(features_path).numpy().squeeze()
            features.append(image_features)

    print("Score path definition")
    scores_path = os.path.join(features_dir, 'scores.npy')

    # Load the scores from the previous session if they exist
    if os.path.exists(scores_path):
        scores = np.load(scores_path)
    else:
        scores = np.ones(len(features))

    display = np.arange(len(features))
    alpha = 0.5

    print("Updating scores")
    scores = UpdateScores(features, scores, display, likeID, alpha)

    # Normalizing score
    scores = scores / np.sum(scores)

    # Save the updated scores for the next session
    np.save(scores_path, scores)

    # Convert the scores to percentages
    scores_pct = ["{:.3f}%".format(score * 100) for score in scores]

    # Combine the filenames and scores
    results = list(zip([str(i).zfill(5) + '.jpg' for i in range(len(features))], scores, scores_pct))

    # Sort the results by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Split the sorted results into separate lists
    filenames, scores, scores_pct = zip(*results)

    # Generate the correct paths for each filename
    filenames = [os.path.join(folder_path, filename) for filename in filenames]
    print(filenames)
    # Combine the filenames and scores again
    filename_score_zip = list(zip(filenames, scores_pct))

    # Pagination
    paginator = Paginator(filename_score_zip, 500) # Show 144 images per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    page_amount = paginator.num_pages

    context = {'filenames': page_obj, 'total_pages': page_amount}
    # TODO Pagination does not work, needs to still be fixed.
    print("Done with Bayesian update.")
    return render(request, 'home.html', context)

# TODO Implement score reset button.
from django.shortcuts import render, redirect
def reset_scores(request):
    print("Resetting scores.npy")
    features_dir = "Features"
    scores_path = os.path.join(features_dir, 'scores.npy')
    # Check if the file exists
    if os.path.exists(scores_path):
        # Delete the file
        os.remove(scores_path)
    # TODO If have time left over fix redirecting issue.
    return redirect('home')

def show_surrounding():
    # TODO Show 50 images before and 50 after the one selected.
    return 0

def search_lion(request):
    query = request.POST.get('query')
    if query is not None:
        print("Query: ", query)
        # Store the query in the session
        request.session['query'] = query
    else:
        # Fetch the query from the session
        query = request.session.get('query', '')

    filenames = []
    similarityExcl = []

    print("Tokenizing text.")
    # Text queries
    text_queries = [query]
    text = tokenizer(text_queries)

    # Directory of images
    image_dir = 'Images'

    # Directory to save image features
    features_dir = 'Features'

    print("Extracting text features.")
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    results = []
    counter = 0

    print("Looping over all images to compare to text features.")
    # Loop over all images in the directory in sorted order
    for image_file in sorted(os.listdir(image_dir)):
        features_path = os.path.join(features_dir, image_file + '.pt')

        # Check if features already exist
        if os.path.exists(features_path):
            # Load the image features from disk
            image_features = torch.load(features_path)
        else:
            # Load and preprocess the image
            image = preprocess(Image.open(os.path.join(image_dir, image_file))).unsqueeze(0)

            # Get the embeddings for the image
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Save the image features to disk
            torch.save(image_features, features_path)

        # Calculate the dot product (similarity) between the text and image embeddings
        similarities = (image_features @ text_features.T).squeeze(0)

        for query, similarity in zip(text_queries, similarities):
            image_path = os.path.join(folder_path, image_file)
            similarity_pct = "{:.3f}".format(similarity * 100) + "%"

            # Store the results
            results.append((image_path, similarity.item(), similarity_pct))

    # Sort the results by similarity in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Split the sorted results into separate lists
    filenames, _, similarityExcl = zip(*results)

    filename_similarity_zip = list(zip(filenames, similarityExcl))

    # Pagination
    paginator = Paginator(filename_similarity_zip, imagesPerPage) # Show 100 images per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    page_amount = paginator.num_pages

    context = {'filenames': page_obj, 'total_pages': page_amount}
    print("Returning results.")
    return render(request, 'home.html', context)
