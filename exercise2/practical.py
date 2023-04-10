#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms


# # Transfer learning on the CIFAR-10 dataset

# In[2]:


# Hyperparameters
learning_rate = 1e-4
batch_size = 50
NUM_EPOCHS = 4  # alter this afterwards
momentum = 0.9
loss_function = nn.CrossEntropyLoss()

# Architecture
NUM_CLASSES = 10


# In[3]:


RESIZE_SIZE = 70
CROP_TO_SIZE = 64

transform = transforms.Compose([
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.RandomCrop((CROP_TO_SIZE, CROP_TO_SIZE)),
    transforms.ToTensor(),
])


cifar_10_training_data = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
cifar_10_test_data = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(
    cifar_10_training_data, batch_size=batch_size, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    cifar_10_test_data, batch_size=batch_size, shuffle=False, num_workers=2
)


# Training-Testing Functions

# In[4]:


writer = SummaryWriter()


# In[5]:


BATCH_TO_PRINT = 100


# In[6]:


def train_model(data_loader, network, optimizer, criterion, num_epochs=NUM_EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    network.train()
    for epoch in range(num_epochs):
        running_loss = epoch_loss = 0.
        for i, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()

            if i % BATCH_TO_PRINT == (BATCH_TO_PRINT - 1):
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / BATCH_TO_PRINT))
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

    print(F'Test accuracy: {(100 * correct / total):.2f}%')


# ## Fine-tuning the model

# In[7]:


alexnet = models.alexnet(weights='AlexNet_Weights.DEFAULT')
alexnet.classifier.add_module('6', nn.Linear(4096, NUM_CLASSES))

optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate, momentum=momentum)


# In[8]:


trained_network = train_model(train_loader, alexnet, optimizer, loss_function, num_epochs=2)
object_to_save = {
    'model_state_dict': trained_network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(object_to_save, 'alexnet-fine-tuned-cifar-10.pt')


# In[ ]:


trained_network = train_model(train_loader, alexnet, optimizer, loss_function, num_epochs=2)
object_to_save = {
    'model_state_dict': trained_network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(object_to_save, 'alexnet-fine-tuned-cifar-10.pt')


# In[9]:


test = test_model(test_loader, trained_network)


# ## Feature extraction

# In[10]:


alexnet = models.alexnet(weights='AlexNet_Weights.DEFAULT')

# Freeze all the layers in the feature extractor
for param in alexnet.features.parameters():
    param.requires_grad = False

alexnet.classifier.add_module('6', nn.Linear(4096, NUM_CLASSES))

optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate, momentum=momentum)
trained_network = train_model(train_loader, alexnet, optimizer, loss_function)
object_to_save = {
    'model_state_dict': trained_network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(object_to_save, 'alexnet-feature-extracted-cifar-10.pt')


# In[11]:


test_model(test_loader, trained_network)

