import os
import wget
import zipfile
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import matplotlib.image as mpimg
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# # Download the train set
# url = 'https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps.zip'
# response = wget.download(url, '../../sources/rps.zip')
# local_file = '../../sources/rps.zip'
# with zipfile.ZipFile(local_file, 'r') as zip_ref:
#     zip_ref.extractall('../../sources/rps-train')
#
# # Download the test set
# url = 'https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps-test-set.zip'
# response = wget.download(url, '../../sources/rps-test-set.zip')
# local_file = '../../sources/rps-test-set.zip'
# with zipfile.ZipFile(local_file, 'r') as zip_ref:
#     zip_ref.extractall('../../sources/rps-test')

base_dir = '../../sources/rps-train/rps'

rock_dir = os.path.join(base_dir, 'rock')
paper_dir = os.path.join(base_dir, 'paper')
scissors_dir = os.path.join(base_dir, 'scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

pic_index = 2
next_rock = [os.path.join(rock_dir, fname) for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    # plt.axis('Off')
    plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
    ])

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

TRAINING_DIR = "../../sources/rps-train/rps"
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255,
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                fill_mode='nearest'
                                                                )
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    target_size=(150, 150),
                                                    batch_size=126,
                                                    class_mode='categorical'
                                                    )

VALIDATION_DIR = "../../sources/rps-test/rps-test-set"
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(150, 150),
                                                              batch_size=126,
                                                              class_mode='categorical'
                                                    )

history = model.fit(train_generator, validation_data=validation_generator, epochs=25)

# Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("JPG files", "*.jpg"), ("All files", "*.*")])
    if file_path:
        selected_file_label.config(text=f"Selected File: {file_path}")
        process_file(file_path)

def process_file(file_path):
    # Implement your file processing logic here
    # For demonstration, let's just display the contents of the selected file
    img = load_img(file_path, target_size=(150, 150))
    x = img_to_array(img)
    print(f'x.shape = {x.shape}')
    x /= 255
    x = np.expand_dims(x, axis=0)
    print(f'x.shape = {x.shape}')

    images = np.vstack([x])
    print(f'images.shape = {images.shape}')
    classes = model.predict(images, batch_size=10)
    print(f'Classes = {classes}')

root = tk.Tk()
root.title("File Dialog Example")

open_button = tk.Button(root, text="Open File", command=open_file_dialog)
open_button.pack(padx=20, pady=20)

selected_file_label = tk.Label(root, text="Selected File:")
selected_file_label.pack()

root.mainloop()

