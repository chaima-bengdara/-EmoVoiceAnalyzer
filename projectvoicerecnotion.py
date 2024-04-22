# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:12:02 2023

@author: chaim
"""

import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load data
data_dir = 'D:/based/test'
target_sample_rate = 22050
alpha = 0.97
num_rows = 4
num_cols = 4

# Mel Frequency Cepstral Coefficients(features extraction)
mfcc_features_list = []

# Labels
labels = []

# LabelEncoder
label_encoder = LabelEncoder()

# Iterate through your audio files
for dirname, _, filenames in os.walk(data_dir):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        # Librosa charges audios after we  can apply pre-emphasis
        audio, _ = librosa.load(file_path, sr=target_sample_rate)
        
        #effect trim to delete the silent from audios 
        audios, _ = librosa.effects.trim(audio, top_db=20)
        
        # spectral Balance: magnitudes will be balanced 

        pre_emphasized_audio = np.append(audios[0], audios[1:] - alpha * audios[:-1])

        # Compute spectrogram
        S = librosa.feature.melspectrogram(y=pre_emphasized_audio, sr=target_sample_rate)
        S_dB = librosa.power_to_db(S, ref=np.max)  # Convert to dB scale

        # Store spectrogram and label
        mfcc_features_list.append(S_dB)
        label = filename.split('_')[-1].split('.')[0].lower()
        labels.append(label)

# Encode labels
labels = label_encoder.fit_transform(labels)

# Normalize features with MinMaxScaler
scaler = MinMaxScaler()

# Normalize each spectrogram feature set
normalized_features = [scaler.fit_transform(feature.T).T for feature in mfcc_features_list]
#(128 time steps,78 features)



# Create subplots for spectrogram visualization
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

# Plot normalized spectrograms in a grid
for i, ax in enumerate(axes.flat):
    if i < len(normalized_features):  # Ensure you don't go out of bounds for the number of audio files
        librosa.display.specshow(normalized_features[i], sr=target_sample_rate, x_axis='time', y_axis='mel', ax=ax)
        ax.set(title=f'Audio {i+1}', xlabel='Time (s)', ylabel='Frequency (Hz)')

# Remove any empty subplots if the number of audio files is less than num_rows * num_cols
for i in range(len(normalized_features), num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])

# Adjust layout
plt.tight_layout()
plt.show()



#model
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Assuming you have already loaded your data and preprocessed it to obtain normalized_features and labels

# Determine Maximum Length and Number of Features
max_length = max([feature.shape[1] for feature in normalized_features])
num_features = normalized_features[0].shape[0]  # Assuming the number of features is the same for all sequences

# Pad Sequences (not needed anymore)
# padded_features = pad_sequences(normalized_features, maxlen=max_length, padding='post', dtype='float32')

# Create Input Data with Masking
X = np.array([np.pad(feature, ((0, 0), (0, max_length - feature.shape[1])), mode='constant') for feature in normalized_features])

# Define the number of classes
num_classes = len(set(labels))

# Convert labels to one-hot encoding
y = to_categorical(labels, num_classes=num_classes)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

lstm_model = Sequential()

# Masking layer to handle variable-length sequences
lstm_model.add(Masking(mask_value=0.0, input_shape=(num_features, max_length)))

# LSTM layer with 128 units and dropout
lstm_model.add(LSTM(128))
lstm_model.add(Dropout(0.7))  # Adding dropout with a dropout rate of 0.2 (you can adjust this value)

# Output layer
lstm_model.add(Dense(num_classes, activation='softmax'))

# Compile the model
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = lstm_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on test data
test_loss, test_accuracy = lstm_model.evaluate(X_test, y_test)

# Print test accuracy
print(f"Test Accuracy: {test_accuracy:.4f}")
lstm_model.summary()
 



import matplotlib.pyplot as plt

# Access the training history

# Plot training & validation loss values

plt.figure(figsize=(15,6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Get the training history
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a range of epochs
epochs = range(1, len(train_accuracy) + 1)

# Plot training and validation accuracy
# Plot accuracy
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, 'r', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_features, labels, test_size=0.2, random_state=42)

# Define a fixed size for your spectrograms
fixed_num_rows = 128  # Set the number of rows to the size you want
fixed_num_cols = 128  # Set the number of columns to the size you want

# Apply padding or truncation to your spectrogram data
X_train = [librosa.util.fix_length(feature, size=fixed_num_cols, axis=1) for feature in X_train]
X_test = [librosa.util.fix_length(feature, size=fixed_num_cols, axis=1) for feature in X_test]

# Convert to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Now, you can train your model using the fixed-size input data

num_classes = len(set(labels))


# Define the CNN model with dropout
cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(fixed_num_rows, fixed_num_cols, 1)))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(64, activation='relu'))
cnn_model.add(layers.Dropout(0.5))  # Adding a dropout layer with dropout rate of 0.5
cnn_model.add(layers.Dense(num_classes, activation='softmax'))

cnn_model.summary()
# Compile the model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = cnn_model.fit(
    np.array(X_train).reshape(-1, fixed_num_rows, fixed_num_cols, 1), 
    np.array(y_train), 
    epochs=10, 
    batch_size=32, 
    validation_data=(np.array(X_test).reshape(-1, fixed_num_rows, fixed_num_cols, 1), np.array(y_test))
)

# Evaluate the model
test_loss, test_acc = cnn_model.evaluate(np.array(X_test).reshape(-1, fixed_num_rows, fixed_num_cols, 1), np.array(y_test))
print(f'Test accuracy: {test_acc}')



# Plot the training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy(CNN)')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy(CNN)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss(CNN)')
plt.plot(history.history['val_loss'], label='Validation Loss(CNN)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout, BatchNormalization

# Define the GRU model
gru_model = models.Sequential()
gru_model.add(GRU(128, input_shape=(fixed_num_rows, fixed_num_cols), return_sequences=True))
gru_model.add(GRU(128, return_sequences=True))
gru_model.add(GRU(128))
gru_model.add(Dropout(0.2))  # Add dropout to reduce overfitting
gru_model.add(BatchNormalization())  # Add batch normalization
gru_model.add(layers.Dense(64, activation='relu'))
gru_model.add(layers.Dense(num_classes, activation='softmax'))

gru_model.summary()

# Compile the model
gru_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = gru_model.fit(
    np.array(X_train).reshape(-1, fixed_num_rows, fixed_num_cols, 1),
    np.array(y_train),
    epochs=10,  # Increase the number of epochs
    batch_size=64,  # Increase the batch size
    validation_data=(np.array(X_test).reshape(-1, fixed_num_rows, fixed_num_cols, 1), np.array(y_test)),
    callbacks=[early_stopping]  # Apply early stopping
)

# Evaluate the model
test_loss, test_acc = gru_model.evaluate(np.array(X_test).reshape(-1, fixed_num_rows, fixed_num_cols, 1), np.array(y_test))
print(f'Test accuracy: {test_acc}')

plt.figure(figsize=(12, 4))

# Plotting training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy (GRU)')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy (GRU)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss (GRU)')
plt.plot(history.history['val_loss'], label='Validation Loss (GRU)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()



# from joblib import dump 
# dump(lstm_model,'C:/Users/chaim/deepldeployment/savedmodels/modellstm.joblib')

# from joblib import dump 
# dump(cnn_model,'C:/Users/chaim/deepldeployment/savedmodels/modelcnn.joblib')

from tensorflow.keras.models import Sequential, load_model

# Assuming you have 'lstm_model' defined

# Save the model using Keras
gru_model.save('C:/Users/chaim/deepldeployment/savedmodels/modelgru.h5')