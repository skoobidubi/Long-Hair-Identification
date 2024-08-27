#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load and preprocess the dataset
# You need a dataset with images, age, gender, and hair length labels
data = pd.read_csv('path_to_your_dataset.csv')

# Preprocess the dataset (assume images are resized and labels are encoded)
X = np.array(data['images'].tolist())
y_age = np.array(data['age'].tolist())
y_gender = np.array(data['gender'].tolist())
y_hair = np.array(data['hair_length'].tolist()) # 1 for long hair, 0 for short hair

# Create a feature for age-based classification
is_age_target = (y_age >= 20) & (y_age <= 30)

# Define a custom function for gender classification
def custom_gender_classification(gender, hair_length, age_range):
    if age_range:  # Age between 20-30
        return 1 if hair_length else 0  # 1: Female, 0: Male
    return gender

y_custom_gender = np.array([custom_gender_classification(g, h, a) for g, h, a in zip(y_gender, y_hair, is_age_target)])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_custom_gender, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification for custom gender
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(f'Model Accuracy: {accuracy_score(y_test, y_pred)}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Save the model
model.save('long_hair_gender_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()


# In[ ]:


import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('long_hair_gender_model.h5')

# Function to load and predict an image
def load_image():
    filepath = filedialog.askopenfilename()
    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (128, 128))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Mock age and hair length for testing purpose
    age = 25  # Assume age is between 20-30 for testing
    hair_length = 1  # Assume long hair for testing

    # Predict gender based on custom conditions
    is_age_target = (age >= 20) & (age <= 30)
    prediction = model.predict(img_array)
    gender = 'Female' if (prediction > 0.5).astype("int32")[0][0] == 1 else 'Male'

    result_label.config(text=f"Predicted Gender: {gender}")

    # Display the image
    img = Image.open(filepath)
    img = img.resize((250, 250))
    imgtk = ImageTk.PhotoImage(img)
    img_label.config(image=imgtk)
    img_label.image = imgtk

# Initialize the GUI window
root = tk.Tk()
root.title("Long Hair Identification for Gender Classification")
root.geometry("400x500")

# Label to display the image
img_label = Label(root)
img_label.pack()

# Label to display the result
result_label = Label(root, text="Upload an image", font=("Arial", 14))
result_label.pack(pady=20)

# Button to upload an image
upload_btn = tk.Button(root, text="Upload Image", command=load_image)
upload_btn.pack(pady=10)

# Run the application
root.mainloop()

