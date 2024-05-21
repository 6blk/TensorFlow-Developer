import os
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import zipfile

import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.utils import load_img, img_to_array

# import wget
#
# URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
# response = wget.download(URL, "../../sources/cats_and_dogs_filtered.zip")
#
# local_file = '../../sources/cats_and_dogs_filtered.zip'
#
# with zipfile.ZipFile(local_file, 'r') as zip_ref:
#     # zip_ref.extractall('../../sources/cats_and_dogs_filtered/')
#     zip_ref.extractall('../../sources/cats_and_dogs_filtered/')


base_dir = '../../sources/cats_and_dogs_filtered'
print('Contents of base directory:')
print(os.listdir(base_dir))

print('\nContents of train directory:')
print(os.listdir(f'{base_dir}/train'))

print('\nContents of validation directory:')
print(os.listdir(f'{base_dir}/validation'))

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

print('Total training cat images: ', len(train_cat_fnames))
print('Total training dog images: ', len(train_dog_fnames))

print('Total validation cat images: ', len(os.listdir(validation_cats_dir)))
print('Total validation dog images: ', len(os.listdir(validation_dogs_dir)))

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8

next_cat_pix = [os.path.join(train_cats_dir, fname) for fname in train_cat_fnames[pic_index-8:pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname) for fname in train_dog_fnames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy']
              )

# All images will be rescaled by 1./255.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    target_size=(150,150),
                                                    class_mode='binary')
valid_generator = valid_datagen.flow_from_directory(validation_dir,
                                                    batch_size=20,
                                                    target_size=(150,150),
                                                    class_mode='binary')

history = model.fit(train_generator, epochs=15, validation_data=valid_generator)

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
    print(f'x.shape = {x.shape}')
    x /= 255
    x = np.expand_dims(x, axis=0)
    print(f'x.shape = {x.shape}')

    images = np.vstack([x])
    print(f'images.shape = {images.shape}')

    classes = model.predict(images, batch_size=10)

    print(f'Classes = {classes}')
    print(classes[0])
    if classes[0] > 0.5:
        print(file_path + " is a dog")
    else:
        print(file_path + " is a cat")

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


# Define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model
successive_outputs = [layer.output for layer in model.layers]
vizualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Prepare a random input image from the training set.
cat_image_file = [os.path.join(train_cats_dir, fname) for fname in train_cat_fnames]
dog_image_file = [os.path.join(train_dogs_dir, fname) for fname in train_dog_fnames]

img_path = random.choice(cat_image_file+dog_image_file)
img = load_img(img_path, target_size=(150, 150))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Scale by 1/255
x /= 255.0

# Run the image through the network, thus obtaining all
# intermediate representations for this image.
successive_feature_map = vizualization_model.predict(x)

# These are the names of the layers, so you can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Display the representations
for layer_name, feature_map in zip(layer_names, successive_feature_map):
    if len(feature_map.shape) == 4:
        # -------------------------------------------
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        # -------------------------------------------
        n_features = feature_map.shape[-1] # number of features in the feature map
        size       = feature_map.shape[ 1] # feature map shape (1, size, size, n_features)

        print(f'n_features = {n_features}')

        # Tile the images in this matrix
        display_grid = np.zeros((size, size * n_features))

        # -------------------------------------------------
        # Postprocess the feature to be visually palatable
        # -------------------------------------------------
        for i in range(n_features):
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *=  64
            x += 128
            x = np.clip(x, 0, 255)#.astype('uint8')
            display_grid[:, i * size : (i+1) * size] = x # Tile each filter into a horizontal grid

        # -----------------
        # Display the grid
        # -----------------
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[    'accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history[     'loss']
val_loss = history.history[ 'val_loss']

epochs = range(len(acc)) # Get number of epochs
#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.show()
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show()