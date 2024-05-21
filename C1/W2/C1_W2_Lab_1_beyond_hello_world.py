import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.6): # Experiment with changing this value
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# You can put between 0 to 59999 here
index = 0

# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'LABEL: {train_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {train_images[index]}')

plt.imshow(train_images[index])
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the classification model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# model.summary()

# Declare sample inputs and convert to a tensor
inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
print(f'Input to softmax function: {inputs}')
inputs = tf.convert_to_tensor(inputs)
print(f'Input to softmax function: {inputs}')
print(f'Input to softmax function: {inputs.numpy()}')

outputs = tf.keras.activations.softmax(inputs)
print(f'Output of softmax function: {outputs.numpy()}')

sum = tf.reduce_sum(outputs)
print(f'Sum of outputs: {sum}')

prediction = np.argmax(outputs)
print(f'Class with highest probability: {prediction}')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=5)

model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])
model.summary()

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])