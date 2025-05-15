import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Uploaded audio files
files = {
    'Shreya_AI': '/content/Shreya_AI.wav',
    'Shreya_likeAI': '/content/Shreya_likeAI.mpeg',
    'Petra_likeAI': '/content/Petra_likeAI.mpeg',
    'Petra_AI': '/content/Petra_AI.wav',
    'Jerusha_like AI': '/content/Jerusha_like AI.mp3',
    'Jerusha_AI': '/content/Jerusha_AI.wav',
    'Anitha_AI': '/content/Anitha_AI.wav',
    'Anitha_ like AI': '/content/Anitha_ like AI.mp3'
}

# Settings
IMG_SIZE = (128, 128)

# Function: Convert audio to spectrogram
def audio_to_spectrogram(filepath):
    y, sr = librosa.load(filepath, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

# Preprocessing
spectrograms = []
labels = []  # 0 = AI, 1 = Human
file_keys = list(files.keys())

for file_key in file_keys:
    spec = audio_to_spectrogram(files[file_key])
    spec_resized = np.resize(spec, IMG_SIZE)
    spectrograms.append(spec_resized)

# Create numpy arrays
X = np.array(spectrograms)[..., np.newaxis]  # Add channel dimension
# Labeling: 'AI' in name (but not 'like AI') -> label 0 (AI), otherwise 1 (Human)
y = np.array([0 if ('AI' in name and 'like' not in name) else 1 for name in file_keys])

# Optional: Visualize spectrograms
for i in range(len(X)):
    plt.figure(figsize=(4, 4))
    plt.imshow(X[i].squeeze(), aspect='auto', origin='lower')
    plt.title(f"Spectrogram of {file_keys[i]}")
    plt.colorbar()
    plt.show()

# Build CNN model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 0 or 1 output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X, y, epochs=50, verbose=2)  # More epochs now

# Predictions
predictions = model.predict(X)
for idx, pred in enumerate(predictions):
    result = 'Human' if pred > 0.5 else 'AI-Generated'
    print(f"{file_keys[idx]} predicted as: {result} (confidence: {pred[0]:.4f})")

# Save model (optional)
model.save('ai_vs_human_voice_classifier_v4.h5')
