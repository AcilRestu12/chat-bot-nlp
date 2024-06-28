import json
import numpy as np
import tensorflow as tf
from nltk_utils import tokenize, stem, bag_of_words
from model import create_model

# Open file json
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Define lists
all_words = []
tags = []
xy = []

# Process each intent
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stemming
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
x_train = []
y_train = []
tag_to_index = {tag: idx for idx, tag in enumerate(tags)}
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    y_train.append(tag_to_index[tag])

# Convert to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Hyperparameters
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000
batch_size = 8

# Create the model
model = create_model(input_size, hidden_size, output_size)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

# Save model and metadata
model.save('chat_model.h5')
metadata = {
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'tags': tags,
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f)
print('Training complete. Model and metadata saved.')
