#!/usr/bin/env python
# coding: utf-8

# In[115]:


import torch.optim as optim
from torchvision import models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter


# In[116]:


# Hyperparameters
learning_rate = 0.0001
batch_size = 50
num_epochs = 4 #alter this afterwards
momentum = 0.9
loss_function = nn.CrossEntropyLoss()

# Architecture
num_classes = 10


# In[117]:



train_transforms = transforms.Compose([transforms.Resize((70, 70)),
                                       transforms.RandomCrop((64, 64)),
                                       transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize((70, 70)),
                                      transforms.CenterCrop((64, 64)),
                                      transforms.ToTensor()])



train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# Training-Testing Functions

# In[118]:


writer = SummaryWriter()


# In[123]:


def train_model(data_loader,network, optimizer, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    network.train()
    for epoch in range(num_epochs):
        running_loss = epoch_loss = 0.
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()

            if i % 100 == 99:
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        writer.add_scalar(f'Loss/train:', epoch_loss / len(data_loader), epoch)
        print(f"[{epoch + 1}] loss: {epoch_loss / len(data_loader):.3f}")
    writer.flush()
    return network

def test_model(data_loader, network):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = alexnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test accuracy: %.2f %%' % (100 * correct / total))


# Fine tuning the model

# In[120]:


# alexnet = models.alexnet(pretrained=True)
alexnet = models.alexnet(weights='AlexNet_Weights.DEFAULT')


# Add a new fully connected layer to the classifier
alexnet.classifier.add_module('6', nn.Linear(4096, 10))


optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate, momentum=momentum)


# In[121]:



# Train the model for a few epochs
trained_net = train_model(train_loader, alexnet,optimizer, loss_function)


# In[ ]:





# In[122]:


# Evaluate the model on the test set

test = test_model(test_loader, trained_net)


# Feature Extraction

# In[126]:


# Load the pre-trained AlexNet model
# alexnet = models.alexnet(pretrained=True)
alexnet = models.alexnet(weights='AlexNet_Weights.DEFAULT')


# Freeze all the layers in the feature extractor
for param in alexnet.features.parameters():
    param.requires_grad = False

# Add a new fully connected layer to the classifier
alexnet.classifier.add_module('6', nn.Linear(4096, 10))

optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate, momentum=momentum)
# Train the model for a few epochs
trained_net = train_model(train_loader,alexnet, optimizer, loss_function)

# Evaluate the model on the test set

test = test_model(test_loader, trained_net)


# In[ ]:




