import tensorflow as tf

# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)
print(dataset)

# Preview the result
for val in dataset:
    print(val.numpy())

dataset = dataset.window(5, 1, drop_remainder=True)
for window_set in dataset:
    print(window_set)

for window_set in dataset:
    print([item.numpy() for item in window_set])

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))
# Print the results
for window in dataset:
  print(window.numpy())

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, 1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
for x, y in dataset:
    print(f'X = {x.numpy()}')
    print(f'Y = {y.numpy()}')
    print()

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5,1,drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
dataset = dataset.shuffle(10)
for x, y in dataset:
    print(f'X = {x.numpy()}')
    print(f'Y = {y.numpy()}')

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5,1,drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
dataset = dataset.shuffle(10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print(f'X = {x.numpy()}')
    print(f'Y = {y.numpy()}')