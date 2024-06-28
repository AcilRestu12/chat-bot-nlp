import random
import json
import numpy as np
import tensorflow as tf
from nltk_utils import bag_of_words, tokenize
from keras.models import load_model

# Load model and metadata
model = load_model('chat_model.h5')
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

input_size = metadata['input_size']
all_words = metadata['all_words']
tags = metadata['tags']

# Open file json
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Chatbot loop
bot_name = 'Tensor'
print("Let's chat! Type 'quit' to exit")

while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break
    
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = np.expand_dims(x, axis=0)
    
    output = model.predict(x, verbose=0)
    predicted = np.argmax(output, axis=1)
    tag = tags[predicted[0]]
    
    probs = tf.nn.softmax(output[0])
    prob = probs[predicted[0]]
    
    threshold = 0.25
    print(f'prob : {prob}')
    if prob > threshold:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}\n")
    else:
        print(f"{bot_name}: I do not understand...\n")
