import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Open file json
with open('intents.json', 'r') as f:
    intents = json.load(f)
    
# Define list
all_words = []  # Setence
tags = []       # Tags of setence
xy = []         # X and Y of setence

# Getting setence, tags, xy
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

# Remove duplicate word
all_words = sorted(set(all_words))      
tags = sorted(set(tags)) 
               
print(tags)

# Define train data
x_train = []
y_train = []
for (pattern_setence, tag) in xy:
    # Getting the feature
    bag = bag_of_words(pattern_setence, all_words)
    x_train.append(bag)
    
    # Getting label from index of tag
    label = tags.index(tag)
    y_train.append(label)   # CrossEntropyLoss
    
# Convert to array
x_train = np.array(x_train)
y_train = np.array(y_train)


# Create custom class dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
        
    def __getItem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
# Define Hyperparameter
batch_size = 8
    
# Create dataset
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    
    
    
    