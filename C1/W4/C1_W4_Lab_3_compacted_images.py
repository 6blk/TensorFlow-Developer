import os
import zipfile
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.utils import load_img, img_to_array

# import wget
#
# URL = "https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip"
# response = wget.download(URL, "../../sources/horse-or-human.zip")

# URL = "https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip"
# response = wget.download(URL, "../../sources/validation-horse-or-human.zip")

# local_file = '../sources/horse-or-human.zip'
#
# with zipfile.ZipFile(local_file, 'r') as zip_ref:
#     zip_ref.extractall('../../sources/horse-or-human/')

# local_file = '../sources/validation-horse-or-human.zip'
#
# with zipfile.ZipFile(local_file, 'r') as zip_ref:
#     zip_ref.extractall('../../sources/validation-horse-or-human/')

# # Unzip training set
# local_zip = '../../sources/horse-or-human.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('../../sources/horse-or-human')
#
# # Unzip validation set
# local_zip = '../../sources/validation-horse-or-human.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('../../sources/validation-horse-or-human')
#
# zip_ref.close()

# Directory with training horse pictures
train_horse_dir = os.path.join('../../sources/horse-or-human/horses')

# Directory with training human pictures
train_human_dir = os.path.join('../../sources/horse-or-human/humans')

# Directory with validation horse pictures
validation_horse_dir = os.path.join('../../sources/validation-horse-or-human/horses')

# Directory with validation human pictures
validation_human_dir = os.path.join('../../sources/validation-horse-or-human/humans')


train_horse_names = os.listdir(train_horse_dir)
print(f'TRAIN SET HORSES: {train_horse_names[:10]}')

train_human_names = os.listdir(train_human_dir)
print(f'TRAIN SET HUMANS: {train_human_names[:10]}')

validation_horse_hames = os.listdir(validation_horse_dir)
print(f'VAL SET HORSES: {validation_horse_hames[:10]}')

validation_human_names = os.listdir(validation_human_dir)
print(f'VAL SET HUMANS: {validation_human_names[:10]}')

print(f'total training horse images: {len(os.listdir(train_horse_dir))}')
print(f'total training human images: {len(os.listdir(train_human_dir))}')
print(f'total validation horse images: {len(os.listdir(validation_horse_dir))}')
print(f'total validation human images: {len(os.listdir(validation_human_dir))}')

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
#     # The fourth convolution (You can uncomment the 4th and 5th conv layers later to see the effect)
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # The fifth convolution
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

# All images will be rescaled by 1./255
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '../../sources/horse-or-human/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=128,
        # Since you used binary_crossentropy loss, you need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '../../sources/validation-horse-or-human/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=32,
        # Since you used binary_crossentropy loss, you need binary labels
        class_mode='binary')

history = model.fit(
    train_generator,
    epochs=15,
    validation_data = validation_generator
)

# history = model.fit(
#       train_generator,
#       steps_per_epoch=8,
#       epochs=15,
#       verbose=1,
#       validation_data = validation_generator,
#       validation_steps=8)

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("Text files", "*.jpg"), ("All files", "*.*")])
    if file_path:
        selected_file_label.config(text=f"Selected File: {file_path}")
        process_file(file_path)

def process_file(file_path):
    # Implement your file processing logic here
    # For demonstration, let's just display the contents of the selected file
    img = load_img(file_path, target_size=(150, 150))
    x = img_to_array(img)
    x /= 255
    print(f'X_Shape = {x.shape}')
    x = np.expand_dims(x, axis=0)
    print(f'X_Shape_V2 = {x.shape}')

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(file_path + " is a human")
    else:
        print(file_path + " is a horse")

    # try:
    #     with open(file_path, 'r') as file:
    #         file_contents = file.read()
    #         file_text.delete('1.0', tk.END)
    #         file_text.insert(tk.END, file_contents)
    # except Exception as e:
    #     selected_file_label.config(text=f"Error: {str(e)}")

root = tk.Tk()
root.title("File Dialog Example")

open_button = tk.Button(root, text="Open File", command=open_file_dialog)
open_button.pack(padx=20, pady=20)

selected_file_label = tk.Label(root, text="Selected File:")
selected_file_label.pack()

# file_text = tk.Text(root, wrap=tk.WORD, height=10, width=40)
# file_text.pack(padx=20, pady=20)

root.mainloop()