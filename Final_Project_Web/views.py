import os
import time

import clip
import cv2
import open_clip
import requests
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from open_clip import tokenizer
from scipy.spatial import distance
from django.shortcuts import render, redirect
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from django.http import HttpResponse

from django.conf import settings
from django.core.paginator import Paginator
import torch
import numpy as np
from sklearn_som.som import SOM

import pickle
import os

folder_path = settings.MEDIA_ROOT

imagesPerPage = 250

# Load the model and tokenizer when the module is loaded
print("Loading model and tokenizer.")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("Loading done.")

@cache_page(60 * 15)
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
        # Append the MEDIA_URL to all filenames to get the URLs
        filename_cluster_zip = [(os.path.join(settings.MEDIA_URL, fn), cluster) for fn, cluster in filename_cluster_zip]
    else:
        # Loop over all feature files in the directory
        for feature_file in sorted(os.listdir(features_dir)):
            # Load the features from disk
            features = torch.load(os.path.join(features_dir, feature_file))

            # Append the features to the list
            all_features.append(features.numpy())

            # Append the corresponding image filename to the filenames list
            image_file = feature_file.replace('.pt', '')
            filenames.append(image_file)
            print("Calculating SOM: ", counter)
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

def send_result(request):
    print("Sent response to server.")
    # key_i = (image_name[-9:])[:5]
    # TODO: Edit object to include selected image.
    my_obj = {'team': "Martin", 'item': "21354"}
    # Query that worked: https://siret.ms.mff.cuni.cz/lokoc/VBSEval/EndPoint.php?team=Martin&item=24563
    # Query to check on: https://siret.ms.mff.cuni.cz/lokoc/VBSEval/eval.php
    response = requests.get(url="https://siret.ms.mff.cuni.cz/lokoc/VBSEval/EndPoint.php", params=my_obj, verify=False)
    print(response)
    return JsonResponse({'result': response.text})

def find_similar(request):
    image_id = request.POST.get('likeID')

    if image_id is not None:
        # Store the image_id in the session
        request.session['image_id'] = image_id
    else:
        # Fetch the image_id from the session
        image_id = request.session.get('image_id', None)

    if image_id is None:
        return HttpResponse("Image ID not found.", status=404)

    # Directory of images
    image_dir = 'Images'

    # Directory to save image features
    features_dir = 'Features'

    # Load the image features from the provided image_id
    image_features_path = os.path.join(features_dir, image_id + '.pt')

    if not os.path.exists(image_features_path):
        return HttpResponse("Image features file not found.", status=404)

    image_features = torch.load(image_features_path)

    results = []

    print("Looping over all images to compare to selected image features.")
    # Loop over all images in the directory in sorted order
    for image_file in sorted(os.listdir(image_dir)):
        features_path = os.path.join(features_dir, image_file + '.pt')

        # Check if features already exist
        if os.path.exists(features_path):
            # Load the image features from disk
            other_image_features = torch.load(features_path)

            # Calculate the dot product (similarity) between the text and image embeddings
            similarity = (image_features * other_image_features).sum()

            # Create the image URL relative to the MEDIA_URL
            image_url = os.path.join(settings.MEDIA_URL, image_file)
            similarity_pct = "{:.3f}".format(similarity * 100) + "%"

            # Store the results
            results.append((image_url, similarity.item(), similarity_pct))

    # Sort the results by similarity in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Split the sorted results into separate lists
    filenames, _, similarityExcl = zip(*results)

    filename_similarity_zip = list(zip(filenames, similarityExcl))

    # Pagination
    paginator = Paginator(filename_similarity_zip, imagesPerPage)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    page_amount = paginator.num_pages

    context = {'filenames': page_obj, 'total_pages': page_amount, 'image_id': image_id}
    print("Returning results.")
    return render(request, 'home.html', context)

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

# TODO: Fix combined clip, does not function like intended.
def combined_clip(request):
    query = request.POST.get('searchClipInput')
    image_id = request.POST.get('likeID')

    if query is not None:
        # Store the query in the session
        request.session['query'] = query
    else:
        # Fetch the query from the session
        query = request.session.get('query', None)

    if image_id is not None:
        # Store the image_id in the session
        request.session['image_id'] = image_id
    else:
        # Fetch the image_id from the session
        image_id = request.session.get('image_id', None)

    if query is None and image_id is None:
        return HttpResponse("Query and Image ID both not found.", status=404)

    filenames = []
    similarityExcl = []

    print("Tokenizing text.")
    # Split the query string on commas to get a list of queries
    text_queries = [q.strip() for q in query.split(',')]
    # Join the queries with the '[SEP]' token
    text_query = ' [SEP] '.join(text_queries)
    print("Query: ", text_query)
    text = tokenizer(text_query)

    # Directory of images
    image_dir = 'Images'

    # Directory to save image features
    features_dir = 'Features'

    print("Extracting text features.")
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Check if an image id was provided
    text_image_similarity = None
    if image_id is not None:
        features_path = os.path.join(features_dir, image_id + '.pt')
        if os.path.exists(features_path):
            image_features = torch.load(features_path)
            print(f"image_features shape: {image_features.shape}")
            print(f"text_features shape: {text_features.shape}")
            text_image_similarity = (image_features * text_features).sum(dim=-1)

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
        similarities = (image_features * text_features).sum(dim=-1)
        if text_image_similarity is not None:
            similarities += text_image_similarity

        for query, similarity in zip(text_queries, similarities):
            # Create the image URL relative to the MEDIA_URL
            image_path = os.path.join(settings.MEDIA_URL, image_file)
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

    context = {'filenames': page_obj, 'total_pages': page_amount, 'query': query}
    print("Returning results.")
    return render(request, 'home.html', context)

def L2S(feature1, feature2):
    return np.sum((feature1 - feature2) ** 2)

def UpdateScores(features, scores, display, likeID, alpha):
    start_time = time.time()

    # Move data to the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert lists of numpy arrays to single numpy arrays
    features = np.array(features)
    scores = np.array(scores)

    # Then, convert numpy arrays to tensors
    features = torch.tensor(features, device=device)
    scores = torch.tensor(scores, device=device)

    # Compute the squared L2 norm between the liked feature and all other features
    diff = features - features[likeID]
    L2_norms = torch.sum(diff ** 2, axis=1)

    # Compute the positive factor (PF)
    PF = torch.exp(-L2_norms / alpha)

    # Compute the negative factor (NF)
    NF = torch.zeros_like(scores)
    for j in display:
        if j != likeID:
            diff = features - features[j]
            L2_norms = torch.sum(diff ** 2, axis=1)
            NF += torch.exp(-L2_norms / alpha)

    # Update the scores
    scores = scores * PF / (NF + 1e-9)  # add a small constant to avoid division by zero

    # Move scores back to the CPU and convert to numpy
    scores = scores.cpu().numpy()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The function took {execution_time} seconds to execute.")

    return scores

def feedback_loop(request):
    # Directory of features
    image_dir = 'Images'
    features_dir = 'Features'
    # Fetch the ID of the liked image from the request
    likeID = request.POST.get('likeID')[:-4]
    print("(Feedback Loop): Image ID, ", likeID)

    likeID = int(likeID)

    if likeID is None:
        likeID = request.session.get('image_id', '')

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
    scores_path = 'scores.npy'

    # Load the scores from the previous session if they exist
    if os.path.exists(scores_path):
        scores = np.load(scores_path)
    else:
        scores = np.ones(len(features))

    display = np.arange(len(features))
    alpha = 0.5

    print("Updating scores")
    scores = UpdateScores(features, scores, display, likeID, alpha)

    print("Normalizing scores")
    # Normalizing score
    scores = scores / np.sum(scores)

    print("Saving old scores")
    # Save the updated scores for the next session
    np.save(scores_path, scores)

    # Convert the scores to percentages
    scores_pct = ["{:.3f}%".format(score * 100) for score in scores]

    # Combine the filenames and scores
    results = list(zip([str(i).zfill(5) + '.jpg' for i in range(len(features))], scores, scores_pct))

    print("Sorting scores")
    # Sort the results by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Split the sorted results into separate lists
    filenames, scores, scores_pct = zip(*results)

    # Generate the correct paths for each filename
    filenames = [os.path.join(settings.MEDIA_URL, filename) for filename in filenames]
    # Combine the filenames and scores again
    filename_score_zip = list(zip(filenames, scores_pct))

    # Pagination
    paginator = Paginator(filename_score_zip, 500) # Show 144 images per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    page_amount = paginator.num_pages

    context = {'filenames': page_obj, 'total_pages': page_amount}
    print("Done with Bayesian update.")
    return render(request, 'home.html', context)

def reset_scores(request):
    print("Resetting scores.npy")
    features_dir = "Features"
    scores_path = 'scores.npy'
    # Check if the file exists
    if os.path.exists(scores_path):
        # Delete the file
        os.remove(scores_path)
    return redirect('home')

def show_surrounding(request):
    image_id = request.POST.get('likeID')

    if image_id is not None:
        # Store the image_id in the session
        request.session['image_id'] = image_id
    else:
        # Fetch the image_id from the session
        image_id = request.session.get('image_id', None)

    if image_id is None:
        return HttpResponse("Image ID not found.", status=404)

    # Directory of images
    image_dir = 'Images'

    # Get all images in sorted order
    all_images = sorted(os.listdir(image_dir))

    try:
        # Find the index of the selected image in the list
        image_index = all_images.index(image_id)
    except ValueError:
        # If the image_id is not found in the list, return an error message
        return HttpResponse('Image not found', status=404)

    # Find the start and end indices for the surrounding images
    start_index = max(0, image_index - 50)
    end_index = min(len(all_images), image_index + 51)

    # Select the surrounding images
    surrounding_images = all_images[start_index:end_index]

    # Create image paths with index
    image_paths_with_index = [(os.path.join(settings.MEDIA_URL, img), idx) for idx, img in enumerate(surrounding_images, start=start_index)]

    # Paginate the images
    paginator = Paginator(image_paths_with_index, imagesPerPage)  # Show 100 images per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    page_amount = paginator.num_pages

    context = {'filenames': page_obj, 'total_pages': page_amount, 'image_id': image_id}
    print("Returning results.")
    return render(request, 'home.html', context)

def search_lion(request):
    query = request.POST.get('query')
    if query is not None:
        # Store the query in the session
        request.session['query'] = query
    else:
        # Fetch the query from the session
        query = request.session.get('query', '')

    filenames = []
    similarityExcl = []

    print("Tokenizing text.")
    # Split the query string on commas to get a list of queries
    text_queries = [q.strip() for q in query.split(',')]
    # Join the queries with the '[SEP]' token
    text_query = ' [SEP] '.join(text_queries)
    print("Query: ", text_query)
    text = tokenizer(text_query)

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
            image_path = os.path.join(settings.MEDIA_URL, image_file)
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

    context = {'filenames': page_obj, 'total_pages': page_amount, 'query': query}
    print("Returning results.")
    return render(request, 'home.html', context)
