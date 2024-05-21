import os
import wget
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt


# url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# response = wget.download(url, '../../sources/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Set the weights file you downloaded into a variable
local_weight_file = '../../sources/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = tf.keras.applications.InceptionV3(input_shape=(150, 150, 3),
                                                      include_top=False,
                                                      weights=None
                                                      )
# pre_trained_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(150, 150, 3),
#                                                                    include_top=False,
#                                                                    weights=None
#                                                                    )

# Load the pre-trained weights you downloaded.
pre_trained_model.load_weights(local_weight_file)

# Freeze the weights of the layers.
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

# Choose `mixed7` as the last layer of your base model
last_layer = pre_trained_model.get_layer('mixed7')
print(f'Last layer output shape = {last_layer.output_shape}')
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = tf.keras.layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = tf.keras.layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Append the dense network to the base model
model = tf.keras.Model(pre_trained_model.input, x)

# Print the model summary. See your dense network connected at the end.
model.summary()

# Set the training parameters
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Download the dataset
# url = 'https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip'
# response = wget.download(url, '../../sources/cats_and_dogs_filtered.zip')

# Extract the archive
# local_file = '../../sources/cats_and_dogs_filtered.zip'
# with zipfile.ZipFile(local_file , 'r') as zip_ref:
#     zip_ref.extractall('../../sources/')

# Define our example directories and files
base_dir = '../../sources/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

print(f'List of Train Dir = {os.listdir(train_dir)}')

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255,
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                zoom_range=0.2,
                                                                shear_range=0.2,
                                                                horizontal_flip=True,
                                                                fill_mode='nearest')
# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0,
#                                                                 rotation_range=40,
#                                                                 shear_range=0.2,
#                                                                 width_shift_range=0.2,
#                                                                 height_shift_range=0.2,
#                                                                 zoom_range=0.2,
#                                                                 horizontal_flip=True)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    class_mode='binary',
                                                    batch_size=20
                                                    )

# Note that the validation data should not be augmented!
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              class_mode='binary',
                                                              batch_size=20
                                                              )

# Train the model.
history = model.fit(train_generator, epochs=20, validation_data=validation_generator)

acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()
