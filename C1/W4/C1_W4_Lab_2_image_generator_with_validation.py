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

validation_horse_names = os.listdir(validation_horse_dir)
print(f'VAL SET HORSES: {validation_horse_names[:10]}')

validation_human_names = os.listdir(validation_human_dir)
print(f'VAL SET HUMANS: {validation_human_names[:10]}')

print(f'Total training horse images: {len(train_horse_names)}')
print(f'Total training human images: {len(train_human_names)}')
print(f'Total validation horse images: {len(validation_horse_names)}')
print(f'Total validation human images: {len(validation_human_names)}')

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0


# Set up matplotlib fig, and size it to fit 4x4 pics
# plt.gcf()
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [ os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [ os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# All images will be rescaled by 1./255
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    '../../sources/horse-or-human',
    target_size=(300,300),
    batch_size=128,
    class_mode='binary'
)

# Flow validation images in batches of 128 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    '../../sources/validation-horse-or-human',
    target_size=(300,300),
    batch_size=32,
    class_mode='binary'
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
)

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("Text files", "*.jpg"), ("All files", "*.*")])
    if file_path:
        selected_file_label.config(text=f"Selected File: {file_path}")
        process_file(file_path)

def process_file(file_path):
    # Implement your file processing logic here
    # For demonstration, let's just display the contents of the selected file
    img = load_img(file_path, target_size=(300, 300))
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

file_text = tk.Text(root, wrap=tk.WORD, height=10, width=40)
file_text.pack(padx=20, pady=20)

root.mainloop()

# # Define a new Model that will take an image as input, and will output
# # intermediate representations for all layers in the previous model after
# # the first.
# successive_outputs = [layer.output for layer in model.layers[1:]]
# visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
#
# # Prepare a random input image from the training set.
# horse_image_files = [os.path.join(train_horse_dir, f) for f  in train_horse_names]
# human_image_files = [os.path.join(train_human_dir, f) for f  in train_human_names]
# img_path = random.choice(horse_image_files+human_image_files)
#
# img = load_img(img_path, target_size=(300, 300)) # this is a PIL image
# x = img_to_array(img)
# x = x.reshape((1,)+x.shape) # Numpy array with shape (1, 300, 300, 3)
#
# # Scale by 1/255
# x /= 255.0
#
# # Run the image through the network, thus obtaining all
# # intermediate representations for this image.
# successive_feature_maps = visualization_model.predict(x)
#
# # These are the names of the layers, so you can have them as part of the plot
# layer_names = [layer.name for layer in model.layers[1:]]
#
# # Display the representations
# for layer_name, feature_map in zip(layer_names, successive_feature_maps):
#     if len(feature_map.shape) == 4:
#         # Just do this for the conv / maxpool layers, not the fully-connected layers
#         n_features = feature_map.shape[-1]  # number of features in feature map
#
#         # The feature map has shape (1, size, size, n_features)
#         size = feature_map.shape[1]
#
#         # Tile the images in this matrix
#         display_grid = np.zeros((size, size * n_features))
#         for i in range(n_features):
#             x = feature_map[0, :, :, i]
#             x -= x.mean()
#             x /= x.std()
#             x *= 64
#             x += 128
#             x = np.clip(x, 0, 255).astype('uint8')
#
#             # Tile each filter into this big horizontal grid
#             display_grid[:, i * size: (i + 1) * size] = x
#
#         # Display the grid
#         scale = 20. / n_features
#         plt.figure(figsize=(scale * n_features, scale))
#         plt.title(layer_name)
#         plt.grid(False)
#         plt.imshow(display_grid, aspect='auto', cmap='viridis')
#         plt.show()