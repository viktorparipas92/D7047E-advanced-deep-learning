#!/usr/bin/env python
# coding: utf-8

# # Task 1 - Train a char-RNN to generate sequences

# In[2]:


import os
import random
import time

import torch
from torch import nn, zeros
from torch.autograd import Variable
from tqdm import tqdm


# In[ ]:


from helpers import (
    ALL_CHARACTERS,
    N_CHARACTERS,
    convert_to_char_tensor,
    read_file,
    time_since,
)


# In[ ]:


DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


# First let's define the char-RNN model based on the `char-rnn.pytorch` repository

# In[ ]:


class CharRNN(nn.Module):
    def __init__(
        self,
        input_size=N_CHARACTERS,
        hidden_size=50,
        output_size=N_CHARACTERS,
        model='gru',
        n_layers=1,
    ):
        super().__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

        if self.model == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)

    def forward(self, input_, hidden):
        encoded = self.encoder(input_)

        batch_size = input_.size(0)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input_, hidden):
        encoded = self.encoder(input_.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden_layer(self, batch_size):
        zeros_matrix = zeros(self.n_layers, batch_size, self.hidden_size)
        if self.model == 'lstm':
            return Variable(zeros_matrix), Variable(zeros_matrix)

        return Variable(zeros_matrix)


# Let's define the function to train the char-RNN model

# In[ ]:


def train_char_rnn(
    model: CharRNN,
    input_,
    target,
    optimizer,
    criterion,
    batch_size=100,
    chunk_length=200,
):
    hidden_layer = model.init_hidden_layer(batch_size)
    hidden_layer.to(DEVICE)

    model.zero_grad()

    loss = 0
    for i in range(chunk_length):
        output_layer, hidden_layer = model(input_[:, i], hidden_layer)
        loss += criterion(output_layer.view(batch_size, -1), target[:, i])

    loss.backward()
    optimizer.step()

    return loss.data[0] / chunk_length


# In[ ]:


def generate(
    model: CharRNN,
    prime_string='A',
    predict_length : int = 100,
    temperature=0.8,
):
    hidden_layer = model.init_hidden_layer(1)
    prime_input = Variable(convert_to_char_tensor(prime_string).unsqueeze(0))

    hidden_layer = hidden_layer.to(DEVICE)
    prime_input = prime_input.to(DEVICE)
    predicted = prime_string

    for i in range(len(prime_string) - 1):
        _, hidden_layer = model(prime_input[:, i], hidden_layer)

    input_ = prime_input[:, -1]
    for p in range(predict_length):
        output, hidden_layer = model(input_, hidden_layer)

        # Sample from the network as a multinomial distribution
        output_distribution = output.data.view(-1).div(temperature).exp()
        top_index = torch.multinomial(output_distribution, 1)[0]

        predicted_char = ALL_CHARACTERS[top_index]
        predicted += predicted_char
        input_ = Variable(convert_to_char_tensor(predicted_char).unsqueeze(0))
        input_.to(DEVICE)

    return predicted


# In[ ]:


def save(model, filename):
    filename = os.path.splitext(
        os.path.basename(filename)
    )[0]
    filename_with_extension = f'{filename}.pt'
    torch.save(model, filename_with_extension)
    print(f'Saved as {filename_with_extension}')


# In[ ]:


def random_training_set(file, file_length, chunk_length, batch_size):
    input_ = torch.LongTensor(batch_size, chunk_length)
    target = torch.LongTensor(batch_size, chunk_length)
    for batch_index in range(batch_size):
        start_index = random.randint(0, file_length - chunk_length)
        end_index = start_index + chunk_length + 1
        chunk = file[start_index : end_index]
        input_[batch_index] = convert_to_char_tensor(chunk[:-1])
        target[batch_index] = convert_to_char_tensor(chunk[1:])

    input_ = Variable(input_).to(DEVICE)
    target = Variable(target).to(DEVICE)

    return input_, target


# In[ ]:


hidden_size = 50
n_layers = 1
model_type = 'lstm'
learning_rate = 0.01
n_epochs = 2000
batch_size = 100
chunk_length = 200

predict_length = 100

filename = 'model'


# In[ ]:


file, file_length = read_file(filename)

decoder = CharRNN(
    hidden_size=hidden_size,
    output_size=N_CHARACTERS,
    model=model_type,
    n_layers=n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_average = 0

try:
    print(f'Training for {n_epochs} epochs...')
    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train_char_rnn(
            decoder,
            *random_training_set(file, file_length, chunk_length, batch_size),
            decoder_optimizer,
            criterion,
            batch_size=batch_size,
            chunk_length=chunk_length,
        )
        loss_average += loss

        if not epoch % 100:
            print(f'[{time_since(start)} ({epoch} {epoch / n_epochs * 100}%) {loss:.4f}]')
            print(f'{generate(decoder, "Wh", predict_length)} \n')

    print('Saving...')
    save(decoder, filename)

