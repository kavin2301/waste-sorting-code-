import librosa
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # You can experiment with different classifiers 
import tempfile
from scipy.io.wavfile import write as wav_write
import sounddevice as sd
import time
import keyboard

# Define mfcc_length globally or pass as an argument
mfcc_length = 100  # Define a fixed length for the MFCC feature vector

# Load audio data and extract features
def load_and_preprocess_data(data_path, mfcc_length=100):  # Set default mfcc_length
    features = []
    labels = []
    for filename in os.listdir(data_path):
        # Load audio file
        y, sr = librosa.load(os.path.join(data_path, filename))

        # Extract features (e.g., MFCCs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # Pad or truncate the MFCC array to a fixed length
        if mfccs.shape[1] < mfcc_length:
            # Pad with zeros if MFCCs are shorter than desired length
            mfccs = np.pad(mfccs, ((0, 0), (0, mfcc_length - mfccs.shape[1])), mode='constant')
        else:
            # Truncate if MFCCs are longer than desired length
            mfccs = mfccs[:, :mfcc_length]

        features.append(mfccs.flatten())  # Flatten to 1D array

        # Extract label from filename (assuming filenames indicate material)
        label = filename.split(".")[0]  # Adjust based on your filename format
        labels.append(label)

    return np.array(features), np.array(labels)

# Load training and testing data
train_features, train_labels = load_and_preprocess_data(r"C:\Users\kavin\OneDrive\Desktop\training", mfcc_length)
test_features, test_labels = load_and_preprocess_data(r"C:\Users\kavin\OneDrive\Desktop\testing", mfcc_length)

# Train the model
model = SVC()  # You can experiment with different model parameters
model.fit(train_features, train_labels)

# Evaluate the model on the test data
accuracy = model.score(test_features, test_labels)
print("Accuracy:", accuracy)
pickle.dump(model, open("your_model.pkl", "wb"))

# Load your trained model (replace with your model loading code)
with open("your_model.pkl", "rb") as f:
    model = pickle.load(f)  # Assuming model is saved as a pickle file

# Train the model with probability=True
model = SVC(probability=True)  # You can experiment with different model parameters
model.fit(train_features, train_labels)

# Set audio capture parameters
duration = 2  # Duration of each audio segment (in seconds)
fs = 22050   # Sampling rate
n_fft = 1024  # Number of samples per frame for MFCC extraction
confidence_threshold = 0.55 # Adjust this threshold based on your needs

# Set the initial start time
start_time = time.time()

while True:
    # Run the process if 5 seconds have elapsed since the last execution
    if time.time() - start_time >= 5:
        if keyboard.is_pressed("w"):
            print("Recording started...")
            record_start_time = time.time()

            data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()

            record_end_time = time.time()
            recording_duration = record_end_time - record_start_time
            print(f"Recording stopped. Duration: {recording_duration:.2f} seconds")
            print("Audio data:", data.flatten())

            if not np.isfinite(data).all():
                print("Error: Audio data contains non-finite values. Skipping feature extraction.")
                continue

            data = data / np.max(np.abs(data))
            mfccs = librosa.feature.mfcc(y=data.flatten(), sr=fs, n_mfcc=20, n_fft=n_fft)

            # Pad or truncate MFCCs
            if mfccs.shape[1] < mfcc_length:
                mfccs = np.pad(mfccs, ((0, 0), (0, mfcc_length - mfccs.shape[1])), mode='constant')
            else:
                mfccs = mfccs[:, :mfcc_length]

            mfccs = mfccs.flatten().reshape(1, -1)

            prediction_proba = model.predict_proba(mfccs)
            if np.max(prediction_proba)>=0.98:  
                predicted_material = "unknown"
            elif np.max(prediction_proba) >= confidence_threshold:
                predicted_material = model.classes_[np.argmax(prediction_proba)]
            else:
                predicted_material = "unknown"

            print("Predicted material:", predicted_material)
            if predicted_material=="plastic":
                predicted_material="p"
            elif predicted_material=="p-tin":
                predicted_material="t"
            else:
                predicted_material="u"
            # ser.write(predicted_material.encode()) # Comment out this line if not using serial communication

            # Update the start time for the next execution
            start_time = time.time()

        # Sleep for a short interval before the next iteration
        time.sleep(0.1)
