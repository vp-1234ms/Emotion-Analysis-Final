from flask import Flask, render_template, request

# Import the functions and models for audio, image, and text processing
from process_functions import process_video, process_audio, process_image
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Process the uploaded file based on the selected type
        file_type = request.form['fileType']
        uploaded_file = request.files['file']

        # Save the uploaded file to a folder
        file_path = f"uploads/{file_type}_{uploaded_file.filename}"
        uploaded_file.save(file_path)

        # Process the file based on its type
        if file_type == 'video':
            predicted_emotions, probabilities=process_video(file_path)
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
