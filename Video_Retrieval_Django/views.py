from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from open_clip import tokenizer
from scipy.spatial import distance
from django.shortcuts import render, redirect
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from django.http import HttpResponse
from scipy.spatial import distance
from sklearn_som.som import SOM
from django.conf import settings
from django.core.paginator import Paginator
from scipy.spatial import distance
from django.core.paginator import Paginator
from django.shortcuts import render
from django.conf import settings

import torch
import numpy as np
import cv2
import numpy as np
import pickle
import os
import os
import cv2
import numpy as np
import pickle
import os
import time
import re
import clip
import cv2
import open_clip
import requests

folder_path = settings.MEDIA_ROOT

imagesPerPage = 250

print("Cache has been cleared before startup.")
cache.clear()
# Load the model and tokenizer when the module is loaded
print("Loading model and tokenizer.")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("Loading done.")

# @cache_page(60 * 15)
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
            print("Calculating SOM: ", counter)
            counter += 1

        # Stack all the feature vectors into a 2D numpy array
        data = np.vstack(all_features)

        # Determine the number of features from the shape of the data array
        n_features = data.shape[1]

        # Initialize a SOM with the correct number of features
        # m = rows, n = columns, big difference in content distribution.
        som = SOM(m=23, n=11, dim=n_features)

        # Fit the SOM to the data and get the cluster assignments
        cluster_assignments = som.fit_predict(data)

        # Create a list of (filename, cluster_assignment) tuples
        filename_cluster_zip = list(zip(filenames, cluster_assignments))

        # Sort the list based on cluster assignments
        filename_cluster_zip.sort(key=lambda x: x[1])

        # Save the cluster assignments for future use
        with open(clusters_file, 'wb') as f:
            pickle.dump(filename_cluster_zip, f)

        filename_cluster_zip = [(os.path.join(settings.MEDIA_URL, fn), cluster) for fn, cluster in filename_cluster_zip]

    # Pagination
    paginator = Paginator(filename_cluster_zip, imagesPerPage)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    page_amount = paginator.num_pages

    context = {'filenames': page_obj, 'total_pages': page_amount}
    return render(request, 'home.html', context)

def send_result(request):
    image_id = request.POST.get('likeID')
    print("Preparing to send response to server. Image: ", image_id)

    if image_id is None:
        return HttpResponse("Image ID not found.", status=404)

    # Remove the last 4 characters from image_id
    image_id = image_id[:-4]

    my_obj = {'team': "Martin", 'item': image_id}

    # Query that worked: https://siret.ms.mff.cuni.cz/lokoc/VBSEval/EndPoint.php?team=Martin&item=24563
    # Query to check on: https://siret.ms.mff.cuni.cz/lokoc/VBSEval/eval.php
    response = requests.get(url="https://siret.ms.mff.cuni.cz/lokoc/VBSEval/EndPoint.php", params=my_obj, verify=False)

    result = parse_response(response.text)
    context = {}
    if result:
        team_name, item_number = result
        print(f"Team name: {team_name}, item number: {item_number}")
        context = {'team_name': team_name, "item_number": item_number}
    else:
        print("Failed to parse response.")

    print("Returning results.")
    return render(request, 'response.html', context)

def parse_response(response_text):
    # Use a regular expression to find the team name and item number
    # This assumes the team name and item number are always in the same position
    match = re.search(r'team name: (.*), submitted item: (.*)', response_text)
    if match:
        team_name = match.group(1)
        item_number = match.group(2)
        return team_name, item_number
    else:
        return None

import os
import torch
from django.conf import settings
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import render

def generate_image_features(image_path):
    # Load the image from the image_path
    image = load_image(image_path)  # Implement this function to load your image

    # Extract features from the image
    features = extract_features(image)  # Implement this function to extract features

    return features

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
        image_path = os.path.join(image_dir, image_id)
        image = Image.open(image_path)
        # convert image to grayscale
        gray_image = image.convert('L')
        # resize image to 128x128
        resized_image = gray_image.resize((128, 128))
        # convert image to numpy array and flatten
        image_features = torch.tensor(np.array(resized_image).flatten())
        image_features = image_features.float() / torch.norm(image_features.float(), 2)

        torch.save(image_features, image_features_path)
    else:
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
        else:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path)
            # convert image to grayscale
            gray_image = image.convert('L')
            # resize image to 128x128
            resized_image = gray_image.resize((128, 128))
            # convert image to numpy array and flatten
            other_image_features = torch.tensor(np.array(resized_image).flatten())
            other_image_features = other_image_features.float() / torch.norm(other_image_features.float(), 2)

            torch.save(other_image_features, features_path)

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


def calculate_histogram(image_file, image_dir, bins):
    # Load the image
    image = cv2.imread(os.path.join(image_dir, image_file))

    # Compute the histogram of the image
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()

    return hist

def search_histogram(request):
    query_image_path = request.POST.get('likeID')
    print("Query image path: ", query_image_path)
    if query_image_path is not None:
        request.session['likeID'] = query_image_path
    else:
        query_image_path = request.session.get('likeID', '')

    query_image_path = os.path.join(os.path.abspath("Images"), query_image_path)
    print("Absolute query image path: ", query_image_path)

    # Number of bins per channel for the histogram
    bins = [8, 8, 8]

    # Load the query image and compute its histogram
    query_image = cv2.imread(query_image_path)
    query_hist = cv2.calcHist([query_image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

    # Normalize the histogram
    query_hist = cv2.normalize(query_hist, query_hist).flatten()

    image_dir = 'Images'
    results = []

    histograms_path = 'histograms.pkl'

    if os.path.exists(histograms_path):
        # Load histograms from file
        with open(histograms_path, 'rb') as f:
            histograms = pickle.load(f)
    else:
        # Calculate histograms and store them in a file
        histograms = {}

        for image_file in sorted(os.listdir(image_dir)):
            hist = calculate_histogram(image_file, image_dir, bins)
            histograms[image_file] = hist

        with open(histograms_path, 'wb') as f:
            pickle.dump(histograms, f)

    for image_file, hist in histograms.items():
        # Compute the cosine similarity between the query image's histogram and this image's histogram
        similarity = 1 - distance.cosine(query_hist, hist)

        image_path = os.path.join(settings.MEDIA_URL, image_file)
        similarity_pct = "{:.3f}".format(similarity * 100) + "%"

        # Store the results
        results.append((image_path, similarity, similarity_pct))

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

    context = {'filenames': page_obj, 'total_pages': page_amount, 'query': query_image_path}
    print("Returning results.")
    return render(request, 'home.html', context)

def combined_clip(request):
    query = request.POST.get('query')
    image_id = request.POST.get('likeID')
    print("Text Query:", query, ", Image Query: ", image_id)

    if query is not None:
        # Store the query in the session
        request.session['query'] = query
    else:
        # Fetch the query from the session
        query = request.session.get('query')

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
    text_query = query.strip()
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

        # Create the image URL relative to the MEDIA_URL
        image_path = os.path.join(settings.MEDIA_URL, image_file)
        similarity_pct = "{:.3f}".format(similarities.item() * 100) + "%"

        # Store the results
        results.append((image_path, similarities.item(), similarity_pct))

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

    context = {'filenames': page_obj, 'total_pages': page_amount, 'query': query}
    print("Returning results.")
    return render(request, 'home.html', context)

def L2S(feature1, feature2):
    return np.sum((feature1 - feature2) ** 2)

def UpdateScores(features, scores, display, likeID, alpha):
    start_time = time.time()

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
            image_features = torch.load(features_path).cpu().numpy().squeeze()
            features.append(image_features)

    print("Score path definition")

    # Load the scores from the session if they exist
    if 'scores' in request.session:
        scores = np.array(request.session['scores'])
    # If the scores aren't found in the session, initialize them to ones
    else:
        scores = np.ones(len(features))

    # Convert lists of numpy arrays to single numpy arrays
    features = np.array(features)
    scores = np.array(scores)

    # Move data to the GPU and convert numpy arrays to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = torch.tensor(features, device=device)
    scores = torch.tensor(scores, device=device)

    display = np.arange(len(features))
    alpha = 0.5

    print("Updating scores")
    scores = UpdateScores(features, scores, display, likeID, alpha)

    print("Normalizing scores")
    # Normalizing score
    scores = scores / torch.sum(scores)

    # Move scores back to the CPU and convert to numpy
    scores_np = scores.cpu().numpy()

    print("Saving old scores")
    # Save the updated scores for the next session in the session
    request.session['scores'] = scores_np.tolist()

    # Convert the scores to percentages
    scores_pct = ["{:.3f}%".format(score * 100) for score in scores_np]

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
    print("Resetting scores in session.")
    print(request.session.keys())  # Print all keys in the session
    if 'scores' in request.session:
        print("Score reset successfully.")
        del request.session['scores']
    else:
        print("'scores' not found in session.")
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
    text_query = query.strip()
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

        image_path = os.path.join(settings.MEDIA_URL, image_file)
        similarity_pct = "{:.3f}".format(similarities.item() * 100) + "%"

        # Store the results
        results.append((image_path, similarities.item(), similarity_pct))

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
