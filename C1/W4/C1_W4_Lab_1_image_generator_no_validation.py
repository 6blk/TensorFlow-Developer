import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import wget
#
# URL = "https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip"
# response = wget.download(URL, "../../sources/horse-or-human.zip")

# Unzip the dataset
# # VARIANT #1
# local_zip = '../../sources/horse-or-human.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('../../sources/horse-or-human/')
# zip_ref.close()

# VARIANT #2
local_zip = '../../sources/horse-or-human.zip'
# with zipfile.ZipFile(local_zip, 'r') as zip_ref:
#     zip_ref.extractall('../../sources/horse-or-human')

# Directory with our training horse pictures
base_dir = '../../sources/horse-or-human'
train_horse_dir = os.path.join('../../sources/horse-or-human/horses')
# Directory with our training human pictures
train_human_dir = os.path.join('../../sources/horse-or-human/humans')
train_horse_names = os.listdir(train_horse_dir)
print(f'{train_horse_names[:10]}')
train_human_names = os.listdir(train_human_dir)
print(f'{train_human_names[:10]}')
print(f'Total number of HORSE training images: {len(train_horse_names)}')
print(f'Total number of HUMAN training images: {len(train_human_names)}')


# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(nrows * 4, ncols * 4)

pic_index += 8

next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate (next_human_pix+next_horse_pix):
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
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

history = model.fit(train_generator, epochs=15, verbose=1)
