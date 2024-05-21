import tensorflow as tf
import numpy as np

def house_model():

    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])

    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=250)

    return model

model = house_model()

print(model.predict([7.0])[0])
