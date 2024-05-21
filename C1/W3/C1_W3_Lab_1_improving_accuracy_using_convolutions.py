import tensorflow as tf
import matplotlib.pyplot as plt

fmnist = tf.keras.datasets.fashion_mnist
(train_data, train_label), (test_data, test_label) = fmnist.load_data()

train_data = train_data /255.0
test_data = test_data / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, 'relu'),
    tf.keras.layers.Dense(10, 'softmax')
])

model.summary()

model.compile('adam', 'sparse_categorical_crossentropy', 'accuracy')

print(f'\nMODEL TRAINING:')
model.fit(train_data, train_label, epochs=5)

print(f'\nMODEL EVALUATION:')
model.evaluate(test_data, test_label)

print(test_label[:100])

f, axarr = plt.subplots(3,4)

FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER =1

layres_outputs = [ layer.output for layer in model.layers]
print(f'LAYRES_OUTPUT = {layres_outputs}')
print(f'LAYRES_INPUT = {model.input}')
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layres_outputs)
activation_model.summary()

for x in range(0, 4):
    f1 = activation_model.predict(test_data[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    if x == 0:
        print(f'F1 = {f1}')
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0, x].grid(False)

    f2 = activation_model.predict(test_data[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)

    f3 = activation_model.predict(test_data[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)

plt.show()