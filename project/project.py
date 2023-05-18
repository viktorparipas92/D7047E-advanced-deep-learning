#!/usr/bin/env python
# coding: utf-8

# # Pneumonia Detection with Chest X-ray Images

# ## Setup

# ### Import libraries

# In[1]:


import sys

import torch
import torchvision
from torchvision.datasets import ImageFolder


# ### Using GPU for training if available

# In[2]:


print(f'Python version: {sys.version_info.major}.{sys.version_info.minor}')
print(f'PyTorch version: {torch.__version__}')
print(f'Torchvision version: {torchvision.__version__}')

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    return device


device = get_device()
print(f'Using device: {device}')


# ## Exploring the dataset

# ### Loading the files

# In[3]:


# Dataset root URL
# https://ltu.app.box.com/s/ywboito9frcx5w4c4mzrrrl4qf2rh9u3/'
ROOT_FOLDER = 'dataset/'
TRAINING_FOLDER = f'{ROOT_FOLDER}train/'
VALIDATION_FOLDER = f'{ROOT_FOLDER}val/'
TEST_FOLDER = f'{ROOT_FOLDER}test/'

training_dataset = ImageFolder(root=TRAINING_FOLDER)
validation_dataset = ImageFolder(root=VALIDATION_FOLDER)
test_dataset = ImageFolder(root=TEST_FOLDER)


# In[ ]:




