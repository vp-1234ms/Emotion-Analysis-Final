<p align="center">
  <a href="" rel="noopener"></a>
</p>
<h1 align="center">Multi-Modal Emotion Detection and Analysis System</h1>

![emotion](https://github.com/vp-1234ms/Emotion-Analysis-Final/assets/102847008/87d94074-8cee-4feb-ad08-992ae57f61a3) <a name = "description"></a>

# 📝 Description <a name = "description"></a>

The Multi-Modal Emotion Detection and Analysis System addresses the contemporary need for a versatile platform capable of interpreting emotions across diverse digital content. Users can seamlessly upload images, audio, or videos, triggering modality-specific processing through specialized models. The system employs a dedicated image emotion detection model for uploaded images, an audio-to-text conversion module for audio files, and a video processing pipeline that extracts frames and converts audio to text for videos. What sets this project apart is its multi-modal fusion approach, harmonizing insights from image, audio, and text models to deliver a holistic emotional profile, particularly crucial for videos where multiple modalities are at play. Standardizing emotions into categories such as Angry, Happy, Sad, Neutral, Fear, and Disgust ensures consistency in analysis. The user interface prioritizes simplicity, guiding users effortlessly through content upload and offering clear, visually appealing emotion analysis reports, including completion pie charts. This user-centric approach extends to scalability, with optimized processing pipelines ensuring efficient analysis, and comprehensive documentation accompanying the system, serving as a valuable resource for users seeking to understand its intricacies. The Multi-Modal Emotion Detection and Analysis System pioneers a nuanced and user-friendly paradigm for decoding emotional content in the digital landscape.

## 💡 Approach <a name = "idea"></a>
Data Integration and Preprocessing:
Utilized various text datasets (ASVP, MEXICAN, RAVDESS, TESS, CREMA, SAVEE), audio dataset, and image dataset.
Integrated these datasets to create a unified text, audio, and image dataset.
Applied data cleaning techniques and performed data sampling, upsampling, and downsampling to balance the dataset.
Data Visualization:
Created bar plots to visualize the frequency count of audio files for each emotion.
Generated pie charts displaying the total count of audio files for each emotion.
Provided a clear understanding of the distribution of emotions in the dataset.
Average Audio Duration:
Machine Learning Model Selection:
Developed code for audio,image and text emotion classification using various machine learning models (Random Forest, Gradient Boosting, SVC, K-Nearest Neighbors, Decision Tree, Naive Bayes, Multi-layer Perceptron, AdaBoost, Quadratic Discriminant Analysis, and XGBoost).
Selected the best-performing model based on accuracy and saved it for future use.
Multimodal Model Training:
Trained various machine learning and custom CNN models for audio data.
Utilized machine learning and custom ANN models for audio and text data.
Integrated Flask for both frontend and backend.
Web Application Integration:
Developed a seamless user experience allowing users to upload image, audio, or video files.
Integrated image, audio, and text models for a comprehensive analysis of video files.
Implemented a result page presenting emotion probabilities in a completion pie chart,bar chart and pe chart sorted from highest to lowest.+-

## ⛏️ Built With <a name = "tech_stack"></a>
- [ASVP,MEXICAN,RAVDESS,TESS,CREMA,SAVEE]-Datasets
- [Data Collection,Data Integration,Data Hadelling,Data Management,Data Processing]-Data Analysis
- [Flask]-Backend Service
- [Python, TensorFlow , Numpy, Pandas , Scikit_Learn,Jupyter Notebook,VsCode , Flask , Keras ,ML , DL(CNN , ANN , Neural Networking]-Machine Learning
- [Matplotlib,Seaborn,Statistics]-Data Visualization
- [HTML,CSS,JS,Bootstrap]-Frontend Service
  
![analysis](https://github.com/vp-1234ms/Emotion-Analysis-Final/assets/102847008/dfb4b7db-f013-493a-8d9d-85798f5f0b1a)

## 🎉 Built By <a name = "acknowledgments"></a>
- Vaibhav Vilas Pawar
- Atharva Gulkotwar
## 🎉 Mentor 
- Dr.Shashank Mouli Satapathy Sir

## Demonstration
https://github.com/vp-1234ms/Multi-Modal-Emotion-Detection-and-Analysis-System/assets/102847008/f8d071f2-2b42-45a7-9511-c14834567b8c

