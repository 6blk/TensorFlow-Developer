import numpy as np
import tensorflow as tf


mnist = tf.keras.datasets.mnist

(x_train,y_train), _ = mnist.load_data()
# print(x_train.shape)
# print(x_train[0][15][16])
# x_train = x_train / 255.0
# print(x_train[0][15][16])
#
# # x_train = np.expand_dims(x_train, axis=-1)
# # x_train = x_train.reshape((60000, 28, 28, -1))
# print(x_train.shape)


# GRADED FUNCTION: reshape_and_normalize

def reshape_and_normalize(images):
    ### START CODE HERE

    # Reshape the images to add an extra dimension
    # Variant #1
    # origin_shape = images.shape
    # new_shape = (*origin_shape, -1)
    # images = images.reshape(images, new_shape)

    # Variant #2
    images = np.expand_dims(images, axis=-1)

    # Normalize pixel values
    images = images / 255.0

    ### END CODE HERE

    return images

training_images = reshape_and_normalize(x_train)
print(f'Maximum pixel value after normalization: {np.max(training_images)}')
print(f'Shape of training set after reshaping: {training_images.shape}')
print(f'Shape of one image after reshaping: {training_images[0].shape}')

# GRADED CLASS: myCallback
### START CODE HERE

# Remember to inherit from the correct class
class myCallback(tf.keras.callbacks.Callback):
    # Define the method that checks the accuracy at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.995):
            print("Reached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True
### END CODE HERE

# GRADED FUNCTION: convolutional_model
def convolutional_model():
    ### START CODE HERE

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3),activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    ### END CODE HERE

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Save your untrained model
model = convolutional_model()

# Instantiate the callback class
callbacks = myCallback()

# Train your model (this can take up to 5 minutes)
history = model.fit(training_images, y_train,epochs=10, callbacks=[callbacks])

print(f"Your model was trained for {len(history.epoch)} epochs")