import tensorflow as tf
import numpy as np
# from tensorflow import keras

print(tf.__version__)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mse')
# model.compile(optimizer='sgd', loss='mean_squared_error')
model.summary()

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

pred = model.predict([10.0])[0]
print(f'SHape of Predict = {pred.shape}')
print(pred)
print(model.predict([10.0]))
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('CPU')))

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
