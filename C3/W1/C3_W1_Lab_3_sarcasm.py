import json
import wget
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# url = 'https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json'
# response = wget.download(url, '../../sources/sarcasm.json')

with open('../../sources/sarcasm.json', 'r') as fl:
    datastore = json.load(fl)

# Non-sarcastic headline
print(datastore[0])

# Sarcastic headline
print(datastore[20000])

# Initialize lists
sentences = []
labels = []
urls = []

# Append elements in the dictionaries into each list
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# Initialize the Tokenizer class
tokenizer = Tokenizer(oov_token='<OOV>')

# Generate the word index dictionary
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(f'number of words in word_index: {len(word_index)}')

# Print the word index
print(f'word_index: {word_index}')
print()

# Generate and pad the sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

# Print a sample headline
index = 2
print(f'Sample headline: {sentences[index]}')
print(f'Padded and tokenized headline= {padded[index]}')

# Print dimensions of padded sequences
print(f'Dim = {padded.shape}')
