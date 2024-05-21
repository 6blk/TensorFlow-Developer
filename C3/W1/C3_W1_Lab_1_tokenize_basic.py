import tensorflow as tf

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!'
]

# Initialize the Tokenizer class
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100)

# Generate indices for each word in the corpus
tokenizer.fit_on_texts(sentences)

# Get the indices and print it
word_index = tokenizer.word_index

print(f'WORD_INDEX = {word_index}')