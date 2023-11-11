import os
import cv2
import moviepy.editor as mp
import speech_recognition as sr
from flask import render_template
import numpy as np
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import librosa
import joblib

# Audio model
audio_loaded_model = load('Final_Audio_Model.joblib')
scaler = joblib.load('audio_scaler.save')
label_encoder = joblib.load("audio_encoder.save")

# Image model
image_loaded_model = load_model('Final_Image_Model.h5')
class_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Text model
text_loaded_model = joblib.load('Final_Text_Model.pkl')
tokenizer = joblib.load('text_tokenizer.save')
max_sequence_length = 150
text_label_encoder = joblib.load('text_label_encoder.save')

# Initialize global variables
predicted_emotions = []
probabilities = []

def process_video(video_path):
    # Task 1: Extract audio
    audio_path = f"uploads/audio_{os.path.basename(video_path)}.wav"
    extract_audio(video_path, audio_path)

    # Task 2: Extract frames
    frames_path = f"uploads/frames_{os.path.basename(video_path)}"
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    extract_frames(video_path, frames_path)

    # Task 3: Extract transcript
    transcript_path = f"uploads/transcript_{os.path.basename(video_path)}.txt"
    extract_transcript(audio_path, transcript_path)

    # Extract audio features from the input audio
    audio_features = extract_features(audio_path)

    if audio_features is not None:
        # Standardize the audio features
        audio_features = scaler.transform([audio_features])

        # Predict the emotion probabilities using the loaded model
        emotion_probabilities = audio_loaded_model.predict_proba(audio_features)
        
        # Get the top 3 emotions with the highest probabilities
        top_emotions_indices = (-emotion_probabilities).argsort()[0][:]
        
        predicted_emotions = label_encoder.inverse_transform(top_emotions_indices)
        probabilities = [emotion_probabilities[0][index] for index in top_emotions_indices]
        
    else:
        audio_emotion_probabilities = np.zeros(3)  # Placeholder for audio probabilities if features are None

    emotion_counts = {emotion: 0 for emotion in class_names}
    total_frames = 0

    for frame_file in os.listdir(frames_path):
        frame_path = os.path.join(frames_path, frame_file)

        # Load and preprocess each frame
        frame_data = load_and_preprocess_image(frame_path)

        # Make predictions on the frame
        image_predictions = image_loaded_model.predict(frame_data)

        # Get the top predicted class label and confidence score
        top_image_class_index = np.argmax(image_predictions[0])
        top_image_class = class_names[top_image_class_index]
        image_confidence = image_predictions[0][top_image_class_index]

        # Update counts
        emotion_counts[top_image_class] += image_confidence
        total_frames += 1

    # Calculate probabilities for each emotion
    emotion_probabilities = {emotion: count / total_frames for emotion, count in emotion_counts.items()}

    # Get the top 3 emotions and their probabilities
    top3_emotions = sorted(emotion_probabilities, key=emotion_probabilities.get, reverse=True)[:]
    top3_probabilities = [emotion_probabilities[emotion] for emotion in top3_emotions]

    with open(transcript_path, 'r') as file:
        text_data = file.read()

    # Tokenize and pad the text data
    text_seq = tokenizer.texts_to_sequences([text_data])
    text_pad = pad_sequences(text_seq, maxlen=max_sequence_length, padding='post')

    # Assuming text_loaded_model is an ExtraTreesClassifier with 50 features
    # Make sure text_pad has the correct shape with 50 features
    text_pad_50_features = text_pad[:, :50]

    # Make a prediction using the loaded model
    predicted_probabilities = text_loaded_model.predict_proba(text_pad_50_features)

    # Get the top three emotions and their probabilities
    top_emotions_indices = np.argsort(predicted_probabilities[0])[:][::-1]
    top_emotions = label_encoder.inverse_transform(top_emotions_indices)
    top_probabilities = predicted_probabilities[0][top_emotions_indices]

    top_probabilities = top_probabilities.tolist()
    predicted_emotions = predicted_emotions.tolist()
    top_emotions=top_emotions.tolist()
    # Make emotions case-insensitive and combine similar emotions
    all_emotions = [emotion.lower() for emotion in predicted_emotions + top3_emotions + top_emotions]
    all_probabilities = probabilities + top3_probabilities + top_probabilities

    # Create a dictionary to associate emotions with their probabilities
    emotion_prob_dict = {}

    for emotion, probability in zip(all_emotions, all_probabilities):
        emotion_prob_dict[emotion] = emotion_prob_dict.get(emotion, 0) + probability

    # Sort the dictionary by probabilities in descending order
    sorted_emotion_prob = {k: v for k, v in sorted(emotion_prob_dict.items(), key=lambda item: item[1], reverse=True)}

    # Extract the top N emotions and their probabilities
    final_top_emotions = list(sorted_emotion_prob.keys())[:]
    final_top_probabilities = list(sorted_emotion_prob.values())[:]


    return final_top_emotions, final_top_probabilities




def process_audio(audio_path):
    # Process audio and get predictions
    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        return None

    transcript_path = f"uploads/transcript_{os.path.basename(audio_path)}.txt"
    extract_transcript(audio_path, transcript_path)

    # Extract audio features from the input audio
    audio_features = extract_features(audio_path)

    if audio_features is not None:
        # Standardize the audio features
        audio_features = scaler.transform([audio_features])

        # Predict the emotion probabilities using the loaded model
        emotion_probabilities = audio_loaded_model.predict_proba(audio_features)
        
        # Get the top 3 emotions with the highest probabilities
        top_emotions_indices = (-emotion_probabilities).argsort()[0][:]
        
        predicted_emotions = label_encoder.inverse_transform(top_emotions_indices)
        probabilities = [emotion_probabilities[0][index] for index in top_emotions_indices]
        
    else:
        audio_emotion_probabilities = np.zeros(3)  # Placeholder for audio probabilities if features are None


    with open(transcript_path, 'r') as file:
        text_data = file.read()

    # Tokenize and pad the text data
    text_seq = tokenizer.texts_to_sequences([text_data])
    text_pad = pad_sequences(text_seq, maxlen=max_sequence_length, padding='post')

    # Assuming text_loaded_model is an ExtraTreesClassifier with 50 features
    # Make sure text_pad has the correct shape with 50 features
    text_pad_50_features = text_pad[:, :50]

    # Make a prediction using the loaded model
    predicted_probabilities = text_loaded_model.predict_proba(text_pad_50_features)

    # Get the top three emotions and their probabilities
    top_emotions_indices = np.argsort(predicted_probabilities[0])[:][::-1]
    top_emotions = label_encoder.inverse_transform(top_emotions_indices)
    top_probabilities = predicted_probabilities[0][top_emotions_indices]

    top_probabilities = top_probabilities.tolist()
    predicted_emotions = predicted_emotions.tolist()
    top_emotions=top_emotions.tolist()
    # Make emotions case-insensitive and combine similar emotions
    all_emotions = [emotion.lower() for emotion in predicted_emotions +  top_emotions]
    all_probabilities = probabilities +  top_probabilities

    # Create a dictionary to associate emotions with their probabilities
    emotion_prob_dict = {}

    for emotion, probability in zip(all_emotions, all_probabilities):
        emotion_prob_dict[emotion] = emotion_prob_dict.get(emotion, 0) + probability

    # Sort the dictionary by probabilities in descending order
    sorted_emotion_prob = {k: v for k, v in sorted(emotion_prob_dict.items(), key=lambda item: item[1], reverse=True)}

    # Extract the top N emotions and their probabilities
    final_top_emotions = list(sorted_emotion_prob.keys())[:]
    final_top_probabilities = list(sorted_emotion_prob.values())[:]


    return final_top_emotions, final_top_probabilities


def process_image(image_path):
    # Load and preprocess the input image
    input_data = load_and_preprocess_image(image_path)

    # Make predictions on the input image
    predictions = image_loaded_model.predict(input_data)

    # Get the top predicted class labels and confidence scores
    top_indices = np.argsort(predictions[0])[::-1][:]
    predicted_emotions = [class_names[i] for i in top_indices]
    probabilities = [predictions[0][i] for i in top_indices]
    # Return the predicted emotions and probabilities
    return predicted_emotions, probabilities


def extract_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])

def extract_frames(video_path, frames_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < max_frames:
        max_frames = frame_count

    frame_interval = frame_count // max_frames

    frame_list = []

    for i in range(0, frame_count, frame_interval):
        cap.set(1, i)
        _, frame = cap.read()
        frame_list.append(frame)

    cap.release()

    for i, frame in enumerate(frame_list):
        cv2.imwrite(os.path.join(frames_path, f"frame_{i+1}.jpg"), frame)

def extract_transcript(audio_path, transcript_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    transcript = recognizer.recognize_google(audio)

    with open(transcript_path, "w") as file:
        file.write(transcript)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
    except Exception as e:
        print(f"Error loading file: {file_path} - {str(e)}")
        return None
    
    if file_path.lower().endswith(".wav"):
        features = [
            np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y)),
            np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
            np.mean(librosa.feature.tempogram(y=y, sr=sr))
        ]
        return features
    else:
        return None

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((80, 80))
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 80, 80, 3)
    return img_array

if __name__ == "__main__":
    # Provide an example video path for testing
    video_path = "example_video.mp4"
    # process_video(video_path)
