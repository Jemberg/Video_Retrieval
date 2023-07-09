# Video Retrieval Course Project

This is a web application written in Python with Django meant for the video retrieval course done in University of Konstanz in the SS of 2023. This application was designed for the purpose of running on a machine remotely and can be accessed via the web or via a VPN by connecting to the host machine.

## Key features:
- Optimized UI for quickly finding the necessary result in the multimedia database.
- Ability to search via a text query for the images, images change once text query has finished.
- Ability to directly submit solution to the API given by the professor.
- Find similar images based on the one selected in the UI.

This application uses ResNET and CLIP to implement the text query and similarity search functions and full credit goes to the authors for these technologies.

## Setup:

Here are the individual steps to set up the project, anyone should be able to do this:
1. Install Required Software: Make sure you have Git, Python, and IntelliJ installed on your machine. You can download Python from the official Python website, Git from the official Git website, and IntelliJ from the official JetBrains website.
2. Clone the Repository: Open your terminal or command prompt, navigate to the directory where you want to clone the repository and type git clone <repository_url>. Replace <repository_url> with the URL of the Git repository.
3. Open the Project in IntelliJ: Launch IntelliJ and from the File menu, select Open. Navigate to the directory where you cloned the Git repository and click OK.
4. Set Up Python Interpreter: After the project has opened, go to File > Settings > Project: <your_project_name> > Python Interpreter. Here, you can add a new interpreter or select an existing one. If the required packages are not installed, IntelliJ will prompt you to install them.
5. Install Django: If Django is not already installed, you can install it using the Python package manager, pip. Open the terminal in IntelliJ (View > Tool Windows > Terminal) and type pip install Django.
6. Start the Django Server: In the IntelliJ terminal, navigate to the directory containing the manage.py file (usually the root directory of the Django project) and type python manage.py runserver. This will start the Django development server and you can access your application by going to http://localhost:8000 in your web browser.
7. Migrate the Database: If you have a database associated with your Django project, you'll need to apply migrations to create the database schema. You can do this by typing python manage.py migrate in the terminal.