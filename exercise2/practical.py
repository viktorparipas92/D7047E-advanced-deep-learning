#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms


# GPU device if possible

# In[2]:


device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


# # Transfer learning on the CIFAR-10 dataset

# In[3]:


# Hyperparameters
learning_rate = 1e-4
batch_size = 50
NUM_EPOCHS = 10
momentum = 0.9
loss_function = nn.CrossEntropyLoss()

# Architecture
NUM_CLASSES = 10


# In[4]:


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

# In[5]:


writer = SummaryWriter()


# In[6]:


BATCH_TO_PRINT = 100


# In[7]:


def train_model(data_loader, network, optimizer, criterion, num_epochs=NUM_EPOCHS):
    network = network.to(device)
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
    network = network.to(device)
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(F'Test accuracy: {(100 * correct / total):.2f}%')


# ## Fine-tuning the model

# In[8]:


alexnet = models.alexnet(weights='AlexNet_Weights.DEFAULT')
alexnet.classifier.add_module('6', nn.Linear(4096, NUM_CLASSES))

optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate, momentum=momentum)


# In[9]:


trained_network = train_model(train_loader, alexnet, optimizer, loss_function)
object_to_save = {
    'model_state_dict': trained_network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(object_to_save, 'alexnet-fine-tuned-cifar-10.pt')


# In[10]:


test = test_model(test_loader, trained_network)


# ## Feature extraction

# In[11]:


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


# In[12]:


test_model(test_loader, trained_network)


# # Transfer learning from MNIST to SVHN

# In[13]:


MNIST_IMAGE_SIZE = 28

num_input_channels = 1
num_output_classes = 10

num_conv1_channels = 32
conv_kernel_size = 3
conv_stride = 1
conv_padding = 1
pool_kernel_size = 2
num_conv2_channels = 64

fc1_output_size = 128

dropout_rate = 0.25

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=num_conv1_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_conv1_channels,
            out_channels=num_conv2_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
        )
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(
            num_conv2_channels * MNIST_IMAGE_SIZE**2 // pool_kernel_size**4,
            fc1_output_size,
        )
        self.fc2 = nn.Linear(fc1_output_size, num_output_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Set up the dataset and data loader

# In[14]:


MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
)
mnist_training_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=mnist_training_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test_data, batch_size=64, shuffle=False)


# Instantiate the model and set up the optimizer and loss function

# In[15]:


learning_rate = 1e-3

my_model = CNN()
optimizer = optim.Adam(my_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# In[16]:


trained_network = train_model(train_loader, my_model, optimizer, loss_function)


# In[17]:


torch.save(trained_network.state_dict(), 'my-cnn-mnist.pt')


# In[18]:


test_model(test_loader, trained_network)


# ## Use pre-trained model for SVNH dataset

# In[41]:


pretrained_model = CNN()
pretrained_model.load_state_dict(torch.load('my-cnn-mnist.pt'))

# Freeze the weights of the pretrained model
for param in pretrained_model.parameters():
    param.requires_grad = False


# Modify the last layer of the model to output 10 classes instead of 2

# In[46]:


pretrained_model.fc1.requires_grad = True

# Modify the first convolutional layer to accept 3 channels instead of 1
pretrained_model.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=num_conv1_channels,
    kernel_size=conv_kernel_size,
    stride=conv_stride,
    padding=conv_padding,
)
SVHN_IMAGE_SIZE = 32
pretrained_model.fc1 = nn.Linear(
    num_conv2_channels * SVHN_IMAGE_SIZE**2 // pool_kernel_size**4,
    fc1_output_size,
)


# Set up the dataset and data loader

# In[47]:


transform = transforms.Compose([
    transforms.Resize((SVHN_IMAGE_SIZE, SVHN_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
svhn_train_data = datasets.SVHN(root='./data', split='train', transform=transform, download=True)
svhn_test_data = datasets.SVHN(root='./data', split='test', transform=transform, download=True)
svhn_train_loader = torch.utils.data.DataLoader(
    dataset=svhn_train_data, batch_size=64, shuffle=True
)
svhn_test_loader = torch.utils.data.DataLoader(
    dataset=svhn_test_data, batch_size=64, shuffle=False
)


# Set up the optimizer and loss function

# In[48]:


optimizer = optim.Adam(pretrained_model.fc2.parameters(), lr=learning_rate)


# In[49]:


trained_network = train_model(
    svhn_train_loader, pretrained_model, optimizer, loss_function, num_epochs=4
)
torch.save(trained_network.state_dict(), 'my-cnn-mnist-pretrained-svhn.pt')


# In[50]:


test_model(svhn_test_loader, trained_network)


# ## Transfer learning

# In[52]:


pretrained_model = CNN()
pretrained_model.load_state_dict(torch.load('my-cnn-mnist.pt'))

pretrained_model.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=num_conv1_channels,
    kernel_size=conv_kernel_size,
    stride=conv_stride,
    padding=conv_padding,
)
SVHN_IMAGE_SIZE = 32
pretrained_model.fc1 = nn.Linear(
    num_conv2_channels * SVHN_IMAGE_SIZE**2 // pool_kernel_size**4,
    fc1_output_size,
)
optimizer = optim.Adam(pretrained_model.fc2.parameters(), lr=learning_rate)
trained_network = train_model(
    svhn_train_loader, pretrained_model, optimizer, loss_function, num_epochs=4
)
torch.save(trained_network.state_dict(), 'my-cnn-mnist-transfer-svhn.pt')


# In[53]:


test_model(svhn_test_loader, trained_network)


# 
