#!/usr/bin/env python
# coding: utf-8

# # Baseline CNN with stochastic gradient descent

# In[8]:


import torch
from torch import nn, no_grad
from torch.nn import Conv2d, CrossEntropyLoss, LeakyReLU, Linear, MaxPool2d
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


# In[9]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

cifar_10_training_data = CIFAR10('datasets/', download=True, transform=transform)
cifar_10_test_data = CIFAR10('datasets/', train=False, download=True, transform=transform)


# In[10]:


train_loader = DataLoader(cifar_10_training_data, batch_size=4, num_workers=2)

test_loader = DataLoader(cifar_10_test_data, batch_size=4, num_workers=2)


# In[24]:


num_input_channels = 3
num_output_classes = 10

num_conv1_channels = 6
conv_kernel_size = 5
pool_kernel_size = 2
num_conv2_channels = 16

fc1_output_size = 120
fc2_output_size = 84


class Net(nn.Module):
    def __init__(self, activation=LeakyReLU, **kwargs):
        super().__init__()

        self.conv1 = Conv2d(
            num_input_channels, num_conv1_channels, conv_kernel_size)
        self.pool1 = MaxPool2d(pool_kernel_size, pool_kernel_size)
        self.conv2 = Conv2d(
            num_conv1_channels, num_conv2_channels, conv_kernel_size)
        self.pool2 = MaxPool2d(pool_kernel_size, pool_kernel_size)
        self.convolution_output_size = num_conv2_channels * conv_kernel_size**2
        # Fully connected layers
        self.fc1 = Linear(
            num_conv2_channels * conv_kernel_size * conv_kernel_size, fc1_output_size)
        self.fc2 = Linear(fc1_output_size, fc2_output_size)
        self.fc3 = Linear(fc2_output_size, num_output_classes)
        self.relu = activation(**kwargs)

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        # Flatten the output of the convolutional layers
        x = x.view(-1, self.convolution_output_size)
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Training the classifier

# In[15]:


def train_model(data_loader, network, optimizer, loss_function):
    for epoch in range(NUMBER_OF_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = network(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % BATCH_TO_PRINT == BATCH_TO_PRINT - 1:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    return network


# In[25]:


net = Net(negative_slope=0.1)


# In[12]:


NUMBER_OF_EPOCHS = 10
BATCH_TO_PRINT = 2000
LEARNING_RATE = 1e-4

criterion = CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=LEARNING_RATE)

# Train the network
trained_net = train_model(train_loader, net, optimizer, criterion)


# Calculate test accuracy

# In[19]:


def calculate_test_accuracy(test_loader, network):
    correct = 0
    total = 0
    with no_grad():
        for data in test_loader:
            images, labels = data
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# In[13]:


accuracy = calculate_test_accuracy(test_loader, net)

print(f'Accuracy of the network on the test images: {100 * accuracy} %')


# The test accuracy is very low, most likely the learning rate is too low since the cross-entropy loss is decreasing very slowly over the epochs. It decreased the same amount during the 10th epoch as the previous nine combined.

# # Swapping the optimizer for ADAM

# In[22]:


from torch.optim import Adam


new_network = Net()
adam_optimizer = Adam(new_network.parameters())

trained_model_with_adam = train_model(train_loader, new_network, adam_optimizer, criterion)


# In[23]:


accuracy = calculate_test_accuracy(test_loader, new_network)

print(f'Accuracy of the network on the test images: {100 * accuracy}%')


# The accuracy is still not very high, but significantly better by only changing the optimizer method from stochastic gradient descent to ADAM.

# # Swapping the activation function for tanh

# In[27]:


from torch.nn import Tanh


network_with_tanh = Net(Tanh)


# In[28]:


adam_optimizer = Adam(network_with_tanh.parameters())
_ = train_model(train_loader, network_with_tanh, adam_optimizer, criterion)


# In[32]:


accuracy = calculate_test_accuracy(test_loader, network_with_tanh)

print(f'Accuracy of the network on the test images: {100 * accuracy:.3f}%')


# The accuracy is lower using the hyperbolical tangent function as activation function in the network, compared to using the leaky ReLU function as activation function.
