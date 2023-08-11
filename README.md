# Video Retrieval Course Project

This is a web application written in Python with Django meant for the video retrieval course done in University of Konstanz in the SS of 2023. This application was designed for the purpose of running on a machine remotely and can be accessed via web hosting or via a VPN by connecting to the host machine.

## Key features:
- Optimized UI for quickly finding the necessary result in the multimedia database, as well as user-error prevention.
- Ability to directly submit solution to the API with a response view.
- Show surrounding images, 50 images before the selected image and 50 after.
- Find similar images based on the pre-computed histograms of other images
- Ability to search via a text query for the images, using Laion2B pre-computed image feature vectors to increase performance.
- A fully implemented Bayesian Feedback Input Loop, which can in multiple steps find close result to multiple inputs.
- Ability to reset Bayesian Feedback Input Loop to start over if result is not satisfactory.
- Combination of text and image query where feature vectors are combined and then searched for.
- Pagination view that allows for seeing all results, as well as proper scrolling interface with sticky menu-bar on top.
- Home page has implemented SOM for image clustering for easier viewing over images.
- Performant caching mechanisms implemented to work perfectly with a CDN or other content distribution providers.

This application uses Laion2B to implement the text query and similarity search functions and full credit goes to the authors for these technologies.

## Setup:

### Prerequisites:
1. Python (3.x or newer) should be installed on your system.
2. pip (Python package manager) should be installed on your system.

### Steps:
1. Clone the repository
Use the command below to clone the repository to your local machine.
`git clone https://git.humbleabo.de/Jemberg/Video_Retrieval_Django`
2. Create a virtual environment (Optional but Recommended)
It's a good practice to create a virtual environment for your Python projects to isolate the project-specific dependencies. You can create a virtual environment using the following command:
`python3 -m venv venv`
Activate the virtual environment with this command on macOS and Linux:
`bash`
`source venv/bin/activate`
For Windows, use:
`.\venv\Scripts\activate`
4. Install the dependencies
The requirements.txt file in the root of the repository contains all the necessary dependencies for the project. Install these using the following command:
`pip install -r requirements.txt`
5. Enter SECRET_KEY, ALLOWED_HOSTS and CSRF_TRUSTED_ORIGINS in settings.py. Do not forget to also put your dataset in a folder named "Images" inside of the root path of the project.
6. Run the server
Now, you should be all set! You can start the Django development server using the following command:
`python manage.py runserver`