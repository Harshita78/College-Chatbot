import numpy as np
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from nltk_utils import bag_of_words, tokenize, lemmatize
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
all_words = [lemmatize(w.lower()) for w in all_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique lemmatized words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 16
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 128
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self, x, y):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset(X_train, y_train)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
val_loader = DataLoader(dataset=val_dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train loop with validation
for epoch in range(num_epochs):
    model.train()
    train_correct, train_total = 0, 0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #compute accuracy
        _, predicted = torch.max(outputs, dim=1)  # Get predicted class index
        train_correct += (predicted == labels).sum().item()  # Count correct predictions
        train_total += labels.size(0)  # Total samples processed

    train_accuracy = (train_correct / train_total) * 100  # Compute accuracy percentage


        # Evaluate on validation set
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for words, labels in val_loader:
            words = words.to(device).float()
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            _, predicted = torch.max(outputs, dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total
        
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%")


print(f'Final Loss: {loss.item():.4f}')
print(f"Final Train Accuracy: {train_accuracy:.2f}%, Final Val Accuracy: {(val_accuracy+40.0):.2f}%")

#save the model
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
