#!/usr/bin/env python
# coding: utf-8

# # Pneumonia Detection with Chest X-ray Images

# ## Setup

# ### Import libraries

# In[1]:


import copy
from itertools import islice
import os
from PIL import Image
import random
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

get_ipython().run_line_magic('matplotlib', 'inline')


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

# ### Find smallest image size

# In[3]:


ROOT_FOLDER = 'dataset/'
TRAINING_FOLDER = f'{ROOT_FOLDER}train/'
VALIDATION_FOLDER = f'{ROOT_FOLDER}val/'
TEST_FOLDER = f'{ROOT_FOLDER}test/'


# In[4]:


def find_smallest_image(dataset_root: str) -> Tuple[str, Tuple[int, int]]:
    smallest_width = smallest_height = None
    smallest_filename = ''
    
    for root, _, filenames in os.walk(dataset_root):
        for filename in filenames:
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root, filename)
                with Image.open(path) as image:
                    width, height = image.size
                    if (
                        smallest_width is None
                        or smallest_height is None
                        or width * height < smallest_width * smallest_height
                    ):
                        smallest_width, smallest_height = (width, height)
                        smallest_filename = filename

    return smallest_filename, (smallest_width, smallest_height)


_, smallest_size_training = find_smallest_image(TRAINING_FOLDER)
_, smallest_size_validation = find_smallest_image(VALIDATION_FOLDER)
_, smallest_size_test = find_smallest_image(TEST_FOLDER)
smallest_width, smallest_height = min(
    smallest_size_training, smallest_size_validation, smallest_size_test
)
print(f'{smallest_width}x{smallest_height}')


# ### Loading and transforming the files

# In[5]:


RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.random.manual_seed(RANDOM_SEED);


# In[6]:


IMAGE_SIZE = 400
ROTATION_DEGREE = 20


# In[7]:


resize = transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE))
to_tensor = transforms.ToTensor()

shared_transforms = [resize, to_tensor]

training_transforms = transforms.Compose([
    resize,
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=ROTATION_DEGREE),
    to_tensor
])


# In[8]:


# Dataset root URL
# https://ltu.app.box.com/s/ywboito9frcx5w4c4mzrrrl4qf2rh9u3/'

training_dataset = ImageFolder(
    root=TRAINING_FOLDER, transform=training_transforms
)
validation_dataset = ImageFolder(
    root=VALIDATION_FOLDER, transform=transforms.Compose(shared_transforms))
test_dataset = ImageFolder(
    root=TEST_FOLDER, transform=transforms.Compose(shared_transforms))


# In[9]:


print(len(training_dataset))
print(len(validation_dataset))
len(test_dataset)


# In[10]:


BATCH_SIZE = 50


# In[11]:


training_loader = DataLoader(
    training_dataset, batch_size=BATCH_SIZE, shuffle=True
)
validation_loader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)


# In[12]:


classes = test_dataset.classes


# In[13]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "\nclass_id = 0\nwhile class_id < len(classes):\n    for images, labels in training_loader:\n        for image, label in zip(images, labels):\n            if class_id == label:\n                plt.figure(figsize=(8, 4))\n                plt.title(f'Class: {classes[class_id]}')\n                plt.imshow(image.permute(1, 2, 0))\n                plt.show()\n                \n                class_id += 1\n")


# ## Training models

# ### Logging

# In[14]:


writer = SummaryWriter()


# In[15]:


def train_model(
    model, criterion, optimizer, training_loader, validation_loader, num_epochs
):
    def train(epoch_loss):
        model.train()
        for _, (images, labels) in enumerate(training_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            training_prediction = model(images)
            training_loss = criterion(training_prediction, labels)
            training_loss.backward()
            
            optimizer.step()

            epoch_loss += training_loss.item() * len(labels)
        
        epoch_loss /= len(training_loader)
        writer.add_scalar(f'Loss/train:', epoch_loss, epoch)
        print(
            f'\r[Training] Epoch [{epoch + 1} / {num_epochs}], '
            f'Epoch Loss: {epoch_loss:.6f}'
        )
        
    def validate():
        model.eval()
        
        with torch.no_grad():
            validation_loss = 0

            for _, (images, labels) in enumerate(validation_loader):
                images = images.to(device)
                labels = labels.to(device)
                validation_prediction = model(images)
                validation_loss += criterion(
                    validation_prediction, labels
                ).item() * len(labels)

            validation_loss /= len(validation_loader)
            writer.add_scalar(f'Loss/train:', validation_loss, epoch)
            print(
                f'\r[Validation] Epoch [{epoch + 1} / {num_epochs}], '
                f'Validation Loss: {validation_loss:.6f}'
            )
    
        return validation_loss
    
    best_validation_loss = None

    model = model.to(device)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        train(epoch_loss)
        validation_loss = validate()

        if best_validation_loss is None or validation_loss < best_validation_loss:
            best_model_state = copy.deepcopy(model.state_dict())
            print('\t Better model found')
    
            best_validation_loss = validation_loss

    writer.flush()
    return best_model_state


# In[16]:


def test_model(model, criterion, test_loader):
    num_correct = 0
    num_total = 0
    test_loss = 0
    true_labels = []
    predicted_labels = []
    
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader):
            labels = labels.to(device)
            num_total += labels.size(0)
            true_labels.extend(labels.tolist())
            
            images = images.to(device)
            prediction = model(images)
            predicted_batch_labels = torch.argmax(prediction, dim=1)
            num_correct += (predicted_batch_labels == labels).sum().item()
            predicted_labels.extend(predicted_batch_labels.tolist())
            
            test_loss += criterion(prediction, labels).item() * len(labels)

    accuracy = num_correct / num_total
    test_loss /= num_total

    print(
        f'\nAccuracy score: {accuracy:.1%} '
        f'({num_correct} correct out of {num_total})')
    
    print(f'Test loss: {test_loss:.4f}')

    return (
        accuracy, 
        test_loss, 
        true_labels, 
        predicted_labels,
    )


# ### Hyperparameters

# In[25]:


NUM_EPOCHS = 10
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-2


# ### Models

# In[18]:


resnet18_model = models.resnet18(weights='DEFAULT')
resnet34_model = models.resnet34(weights='DEFAULT')
alexnet = models.alexnet(weights='DEFAULT')

models_to_fine_tune = {
    'ResNet-18': resnet18_model,
    # 'ResNet-34': resnet34_model,
    # 'AlexNet': alexnet,
}


# In[19]:


NUM_MODELS_TO_SKIP = 0


# In[20]:


models_to_train = dict(
    islice(models_to_fine_tune.items(), NUM_MODELS_TO_SKIP, None)
)
list(models_to_train.keys())


# In[26]:


num_classes = len(classes)


for model_name, model in models_to_train.items():
    if model_name.startswith('ResNet'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'AlexNet':
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes
        )
    
    try:
        best_model_filename = f'best-model-{model_name}.pth'
        best_model_state = torch.load(best_model_filename)
        torch.load(best_model_state, f'best-model-{model_name}.pth')
        model.load_state_dict(best_model_state)
    except Exception:
        pass
    
    optimizer_for_fine_tuning = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
    )
    print(f'Training {model_name}: \n')
    
    best_fine_tuned_model_state = train_model(
        model, 
        nn.CrossEntropyLoss(), 
        optimizer_for_fine_tuning, 
        training_loader, 
        validation_loader, 
        NUM_EPOCHS,
    )
    torch.save(best_fine_tuned_model_state, f'best-model-{model_name}.pth')


# ## Evaluating the models

# ### Loading the best models

# In[27]:


criterion_ft = nn.CrossEntropyLoss()

best_models = {}
for model_name, model in models_to_fine_tune.items():
    try:
        best_model_filename = f'best-model-{model_name}.pth'
        best_model_state = torch.load(best_model_filename)
        if model_name.startswith('ResNet'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'AlexNet':
            model.classifier[6] = nn.Linear(
                model.classifier[6].in_features, num_classes
            )
        
        model.load_state_dict(best_model_state)
    except Exception as e:
        print(f'Could not load model {model_name}: {e}')
    
    best_models[model_name] = model

list(best_models.keys())


# ### Evaluating performance

# In[28]:


def plot_confusion_matrix_heatmap(
    confusion_matrix: pd.DataFrame, model_name: str
):
    heatmap = sns.heatmap(confusion_matrix_dataframe, annot=True)
    plt.ylabel('True label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=14, fontweight='bold')
    plt.title(f'Confusion matrix for {model_name}')
    plt.show()

    
def plot_mislabeled_images(
    test_loader, y_predicted, y_true, classes=classes
):
    for (images, labels) in test_loader:
        for i, (predicted_label, true_label, image, label) in enumerate(
            zip(y_predicted, y_true, images, labels)
        ):
            if predicted_label != true_label:
                plt.figure(figsize=(8, 4))
                plt.title(
                    f'#{i} Predicted Label: "{classes[predicted_label]}" '
                    f'True Label: "{classes[true_label]}" '
                )
                plt.imshow(image.permute(1, 2, 0))
                plt.show()


# In[29]:


test_loader = DataLoader(
    test_dataset, batch_size=50, shuffle=True
)


for model_name, model in best_models.items():
    print(f'\nEvaluating model {model_name}:')
    
    _, _, y_true, y_predicted = test_model(
        model, criterion_ft, test_loader
    )
    confusion_matrix_ = confusion_matrix(y_true, y_predicted)
    confusion_matrix_dataframe = pd.DataFrame(
        confusion_matrix_, 
        index=classes,
        columns=classes,
    )

    plot_confusion_matrix_heatmap(
        confusion_matrix_dataframe, model_name
    )
    
    # plot_mislabeled_images(
    #     test_loader, y_predicted, y_true, classes
    # )


# In[ ]:




