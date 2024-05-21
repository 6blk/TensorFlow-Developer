import os
import wget
import tensorflow as tf
import matplotlib.pyplot as plt


# import wget
#
# URL = "https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip"
# response = wget.download(URL, "../../sources/horse-or-human.zip")

# URL = "https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip"
# response = wget.download(URL, "../../sources/validation-horse-or-human.zip")

#
# local_file = '../../sources/horse-or-human.zip'
#
# with zipfile.ZipFile(local_file, 'r') as zip_ref:
#     zip_ref.extractall('../../sources/horse-or-human/')

# local_file = '../../sources/validation-horse-or-human.zip'
# with zipfile.ZipFile(local_file, 'r') as zip_ref:
#     zip_ref.extractall('../../sources/validation-horse-or-human/')

# Directory with training horse pictures
train_base_dir = '../../sources/horse-or-human'
validation_base_dir = '../../sources/validation-horse-or-human'

train_horse_dir = os.path.join(train_base_dir, 'horses')

# Directory with training human pictures
train_human_dir = os.path.join(train_base_dir, 'humans')

# Directory with validation horse pictures
validation_horse_dir = os.path.join(validation_base_dir, 'horses')

# Directory with validation human pictures
validation_human_dir = os.path.join(validation_base_dir, 'humans')

print(f'Number of files in Validation Horse Directory = {len(os.listdir(validation_horse_dir))}')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
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



model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

model.summary()

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0,
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                fill_mode='nearest'
                                                                )
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(train_base_dir,
                                                    batch_size=128,
                                                    target_size=(300, 300),
                                                    class_mode='binary'
                                                    )

validation_generator = validation_datagen.flow_from_directory(validation_base_dir,
                                                              batch_size=32,
                                                              target_size=(300, 300),
                                                              class_mode='binary'
                                                              )

history = model.fit(train_generator, epochs=20, validation_data=validation_generator)

acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.show()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
