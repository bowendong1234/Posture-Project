import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
from CombinedDataGenerator import CombinedDataGenerator

# Directory paths for good and bad posture images
good_posture_dir = r'C:\Users\Bowen\OneDrive\Documents\Engineering and Computer Science\Personal Projects\good_posture_pics'
too_close_to_screen_dir = r'C:\Users\Bowen\OneDrive\Documents\Engineering and Computer Science\Personal Projects\bad_posture_pics'

# Image dimensions and batch size
img_width, img_height = 960, 540
batch_size = 8

# Get the total number of images for the model
count_good_posture_images = len(os.listdir(r'C:\Users\Bowen\OneDrive\Documents\Engineering and Computer Science\Personal Projects\good_posture_pics\good_posture'))
count_too_close_to_screen_images = len(os.listdir(r'C:\Users\Bowen\OneDrive\Documents\Engineering and Computer Science\Personal Projects\bad_posture_pics\too_close_to_screen'))
total_samples = count_good_posture_images + count_too_close_to_screen_images


# Data preprocessing and augmentation for good posture images
good_datagen = image.ImageDataGenerator(rescale=1./255)
good_generator = good_datagen.flow_from_directory(
    directory=good_posture_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Data preprocessing and augmentation for too close to screen images
bad_datagen = image.ImageDataGenerator(rescale=1./255)
bad_generator = bad_datagen.flow_from_directory(
    directory= too_close_to_screen_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)


# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("FUCKKKKK")
print(count_good_posture_images // batch_size)
# Train the model
history = model.fit(
    good_generator,
    steps_per_epoch=count_good_posture_images // batch_size,
    epochs=10,
    validation_data=bad_generator,
    validation_steps=count_too_close_to_screen_images // batch_size
)

# Save the trained model
model.save('posture_detection_model.h5')