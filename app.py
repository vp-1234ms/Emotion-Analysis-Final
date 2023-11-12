from flask import Flask, render_template, request
import atexit
import shutil
import os

# Import the functions and models for audio, image, and text processing
from process_functions import process_video, process_audio, process_image

app = Flask(__name__)

# Path to the 'uploads' folder
uploads_folder = 'uploads'

# Create the 'uploads' folder if it doesn't exist
if not os.path.exists(uploads_folder):
    os.makedirs(uploads_folder)
    print(f"Created {uploads_folder} folder.")

def cleanup():
    # Delete the 'uploads' folder and its contents
    if os.path.exists(uploads_folder):
        shutil.rmtree(uploads_folder)
        print(f"Deleted {uploads_folder} folder.")

# Register the cleanup function to be called when the application exits
atexit.register(cleanup)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Process the uploaded file based on the selected type
        file_type = request.form['fileType']
        uploaded_file = request.files['file']

        # Save the uploaded file to a folder
        file_path = f"{uploads_folder}/{file_type}_{uploaded_file.filename}"
        uploaded_file.save(file_path)

        # Process the file based on its type
        if file_type == 'video':
            predicted_emotions, probabilities = process_video(file_path)
            print("Hi APP VIDEO")
            print(predicted_emotions)
            print(probabilities)
        elif file_type == 'audio':
            predicted_emotions, probabilities = process_audio(file_path)
            print("Hi APP AUDIO")
            print(predicted_emotions)
            print(probabilities)
        elif file_type == 'image':
            predicted_emotions, probabilities = process_image(file_path)
            print("Hi APP IMAGE")
            print(predicted_emotions)
            print(probabilities)

        # Display the results on a new webpage
        return render_template('results.html', emotions=predicted_emotions, probabilities=probabilities)

    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
