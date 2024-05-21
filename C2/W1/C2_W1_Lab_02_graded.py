import os.path
import random
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt

import wget
import zipfile

#URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
#response = wget.download(URL, "../../sources/cats-and-dogs.zip")

#local_file = '../../sources/cats-and-dogs.zip'

#with zipfile.ZipFile(local_file, 'r') as zip_ref:
#    #zip_ref.extractall('../../sources/')
#    zip_ref.extractall()

def filter_files(directory, extension_to_keep):
    for dir_path, dir_name, file_name in os.walk(directory):
        for file in file_name:
            # DELETE files of unsupported formats
            if not file.endswith(extension_to_keep):
                print(f'Delete file: {os.path.join(dir_path, file)}')
                os.remove(os.path.join(dir_path, file))

            # Delete empty files
            if  os.path.getsize(os.path.join(dir_path, file)) == 0:
                print(f'Delete empty file: {file}')
                print(f'{file} is zero length, so ignoring.')
                os.remove(os.path.join(dir_path, file))


source_path = '../../sources/PetImages'

source_path_dogs = os.path.join(source_path, 'Dog')
source_path_cats = os.path.join(source_path, 'Cat')

# # Deletes all non-image files (there are two .db files bundled into the dataset)
extension_to_keep = '.jpg'

filter_files(source_path, extension_to_keep)

# os.listdir returns a list containing all files under the given path
print(f"There are {len(os.listdir(source_path_dogs))} images of dogs.")
print(f"There are {len(os.listdir(source_path_cats))} images of cats.")

# Define root directory
root_dir = '../../sources/cats-v-dogs'

# # Empty directory to prevent FileExistsError is the function is run several times
# if os.path.exists(root_dir):
#     shutil.rmtree(root_dir)

def create_train_val_dirs(root_path):
    os.makedirs(os.path.join(root_path, 'training', 'cats'))
    os.makedirs(os.path.join(root_path, 'training', 'dogs'))
    os.makedirs(os.path.join(root_path, 'validation', 'cats'))
    os.makedirs(os.path.join(root_path, 'validation', 'dogs'))

# try:
#   create_train_val_dirs(root_path=root_dir)
# except FileExistsError:
#   print("You should not be seeing this since the upper directory is removed beforehand")

# Test your create_train_val_dirs function
for rootdir, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        print(os.path.join(rootdir, subdir))

def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
    files = os.listdir(SOURCE_DIR)
    randomized_list = random.sample(files, len(files))

    for fl in randomized_list[:int(SPLIT_SIZE * len(randomized_list))]:
        #print(f'Array len = {len(randomized_list[:int(SPLIT_SIZE * len(randomized_list))])}')
        # print(f'File Path = {os.path.join(TRAINING_DIR, fl)}')
        shutil.copyfile(os.path.join(SOURCE_DIR, fl), os.path.join(TRAINING_DIR, fl))
    for fl in randomized_list[int(SPLIT_SIZE * len(randomized_list)):]:
        #print(f'Array len = {len(randomized_list[:int(SPLIT_SIZE * len(randomized_list))])}')
        #print(f'Array len = {len(randomized_list[int(SPLIT_SIZE * len(randomized_list)):])}')
        #print(f'File Path = {os.path.join(VALIDATION_DIR, fl)}')
        shutil.copyfile(os.path.join(SOURCE_DIR, fl), os.path.join(VALIDATION_DIR, fl))

CAT_SOURCE_DIR = '../../sources/PetImages/Cat'
DOG_SOURCE_DIR = '../../sources/PetImages/Dog'

TRAINING_DIR    = '../../sources/cats-v-dogs/training'
VALIDATION_DIR  = '../../sources/cats-v-dogs/validation'

TRAINING_CATS_DIR   = os.path.join(TRAINING_DIR, 'cats')
VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, 'cats')

TRAINING_DOGS_DIR   = os.path.join(TRAINING_DIR, 'dogs')
VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, 'dogs')

# # Empty directories in case you run this cell multiple times
# print(f'TRAINING_CATS_DIR_LEN = {len(os.listdir(TRAINING_CATS_DIR))}')
# if len(os.listdir(TRAINING_CATS_DIR)) > 0:
#     for file in os.scandir(TRAINING_CATS_DIR):
#         print(f'file = {file}')
#         print(f'file.path = {file.path}')
#         #os.remove(file.path)
# if len(os.listdir(VALIDATION_CATS_DIR)) > 0:
#     for file in os.scandir(VALIDATION_CATS_DIR):
#         print(f'file = {file}')
#         print(f'file.path = {file.path}')
#         #os.remove(file.path)
# if len(os.listdir(TRAINING_DOGS_DIR)) > 0:
#     for file in os.scandir(TRAINING_DOGS_DIR):
#         print(f'file = {file}')
#         print(f'file.path = {file.path}')
#         #os.remove(file.path)
# if len(os.listdir(VALIDATION_DOGS_DIR)) > 0:
#     for file in os.scandir(VALIDATION_DOGS_DIR):
#         print(f'file = {file}')
#         print(f'file.path = {file.path}')
#         # os.remove(file.path)

# Define proportion of images used for training
split_size = 0.9

# # Run the function
# # NOTE: Messages about zero length images should be printed out
# split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, split_size)
# split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, split_size)

# Check that the number of images matches the expected output
print(f"\n\nOriginal cat's directory has {len(os.listdir(CAT_SOURCE_DIR))} images")
print(f"Original dog's directory has {len(os.listdir(DOG_SOURCE_DIR))} images\n")

# Training and validation splits
print(f"There are {len(os.listdir(TRAINING_CATS_DIR))} images of cats for training")
print(f"There are {len(os.listdir(TRAINING_DOGS_DIR))} images of dogs for training")
print(f"There are {len(os.listdir(VALIDATION_CATS_DIR))} images of cats for validation")
print(f"There are {len(os.listdir(VALIDATION_DOGS_DIR))} images of dogs for validation")

# GRADED FUNCTION: train_val_generators
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
    """
    Creates the training and validation data generators

    Args:
      TRAINING_DIR (string): directory path containing the training images
      VALIDATION_DIR (string): directory path containing the testing/validation images

    Returns:
      train_generator, validation_generator - tuple containing the generators
    """
    # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        target_size=(150, 150),
                                                        batch_size=128,
                                                        class_mode='binary'
                                                        )

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                    target_size=(150, 150),
                                                    batch_size=25,
                                                    class_mode='binary'
                                                    )
    return train_generator, validation_generator

# Test your generators
train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

def creat_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )
    return model

model = creat_model()

model.summary()

# Train the model
# Note that this may take some time.
history = model.fit(train_generator, epochs=15, validation_data=validation_generator)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()
print("")

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()
