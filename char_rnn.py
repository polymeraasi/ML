"""
Character-level Recurrent Neural Network (RNN)

This script implements a simple character-level RNN using PyTorch. The model learns to generate text
based on a given input sequence by predicting the next character.

Main Components:
- V3_RNN: A simple recurrent neural network that maintains hidden states and predicts the next character.
- Encoder: Converts a text file into a one-hot encoded matrix and character mappings.
- Training loop: Optimizes the model using cross-entropy loss.
- Evaluation: Generates text sequences from a random start character or a fixed sequence.

"""


import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class V3_RNN(nn.Module):
    """
    A simple character-level recurrent neural network.
    
    Parameters:
        - input_size: Number of unique characters (vocabulary size).
        - hidden_size: Number of hidden units in the recurrent layer.
        - output_size: Same as input_size (character prediction).
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(V3_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.f1 = nn.Linear(input_size + hidden_size, hidden_size)  # update state
        self.f2 = nn.Linear(hidden_size, output_size)  # predict output
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)  # combined current input and previous state
        new_state = torch.tanh(self.f1(combined))
        output = self.softmax(self.f2(new_state))
        return output, new_state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


def encoder(file_name):
    """
    Converts a text file into a one-hot encoded matrix and provides character mappings.
    
    Returns:
        - One-hot encoded matrix
        - Vocabulary size
        - Character-to-integer dictionary
        - Integer-to-character dictionary
    """
     
    char_to_int = {}
    with open(file_name, 'r') as file:
        txt = file.read()

    for char in txt:
        if char not in char_to_int:
            char_to_int[char] = len(char_to_int)

    onehot_matrix = np.zeros((len(txt), len(char_to_int)), dtype=int)
    for i, char in enumerate(txt):
        onehot_matrix[i, char_to_int[char]] = 1   # [row, col]

    int_to_char = {i: char for char, i in char_to_int.items()}
    vocab_size = len(char_to_int)

    return onehot_matrix, vocab_size, char_to_int, int_to_char


txt_file = 'abcde.txt'
#txt_file = 'abcde_edcba.txt'
onehot_matrix, vocabulary_size, char_2_int, int_2_char = encoder(txt_file)
onehot_matrix = onehot_matrix[:1000]  # reducing the amount of training data

tot_chars = len(onehot_matrix)
x = []
y = []
for i in range(tot_chars - 1):
    x.append(onehot_matrix[i:i+1])
    y.append(np.argmax(onehot_matrix[i+1]))

x_array = np.array(x)
x_tensor = torch.tensor(x_array, dtype=torch.float32).squeeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

hidden_size = 128
model = V3_RNN(vocabulary_size, hidden_size, vocabulary_size)


def train(model, dataloader):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_data, y_data in dataloader:
            state = model.init_hidden(x_data.size(0))
            optimizer.zero_grad()
            predictions, state = model(x_data, state)
            loss = loss_function(predictions, y_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1} / {num_epochs}, Loss: {total_loss / len(dataloader)}')


train(model, dataloader)


def evaluate(model, start, int_to_char, char_to_int, vocab_size):
    """
    Generates a sequence of characters starting from a single given character.
    
    Parameters:
        - model: The trained RNN model.
        - start: The starting character for generation.
        - int_to_char: Mapping from integer indices to characters.
        - char_to_int: Mapping from characters to integer indices.
        - vocab_size: The total number of unique characters in the dataset.
    
    Returns:
        - A generated text sequence of length 50 starting from the given character.
    """
    length_of_seq = 50
    model.eval()
    input_i = torch.tensor([[char_to_int[start]]], dtype=torch.long)
    input = F.one_hot(input_i, num_classes=vocab_size).float().squeeze(0)
    state = model.init_hidden(1)

    generated_txt = start
    with torch.no_grad():
        for _ in range(length_of_seq - 1):
            output, state = model(input, state)
            next_char_i = torch.argmax(output, dim=1)
            next_char = int_to_char[next_char_i.item()]
            generated_txt += next_char
            input = F.one_hot(torch.tensor([[next_char_i.item()]]), num_classes=vocab_size).float().squeeze(0)

    return generated_txt


def generate_from_sequence(model, fixed_sequence, int_to_char, char_to_int, vocab_size):
    """
    Generates a sequence of characters starting from a given fixed sequence.
    
    Parameters:
        - model: The trained RNN model.
        - fixed_sequence: The starting string for text generation.
        - int_to_char: Mapping from integer indices to characters.
        - char_to_int: Mapping from characters to integer indices.
        - vocab_size: The total number of unique characters in the dataset.
    
    Returns:
        - A generated text sequence of length 50, continuing from the fixed sequence.
    """
    length_of_seq = 50
    model.eval()
    fixed_seq_i = [char_to_int[char] for char in fixed_sequence]
    input_tensor = torch.tensor(fixed_seq_i, dtype=torch.long)
    input_onehot = F.one_hot(input_tensor, num_classes=vocab_size).float().squeeze(0)
    state = model.init_hidden(1)

    generated_txt = fixed_sequence
    with torch.no_grad():
        for _ in range(length_of_seq - len(fixed_sequence)):
            output, state = model(input_onehot[-1].unsqueeze(0), state)
            next_char_i = torch.argmax(output, dim=1)
            next_char = int_to_char[next_char_i.item()]
            generated_txt += next_char
            input_onehot = torch.cat((input_onehot, F.one_hot(next_char_i, num_classes=vocab_size).float()), dim=0)

    return generated_txt


# for evaluation, a random start char and a fixed sequence
start_char = random.choice(list(int_2_char.values()))
generated = evaluate(model, start_char, int_2_char, char_2_int, vocabulary_size)
print("Generated sequence (from random start):", generated)

fixed_sequence = 'abc'
generated_fixed = generate_from_sequence(model, fixed_sequence, int_2_char, char_2_int, vocabulary_size)
print("Generated sequence (from sequence):", generated_fixed)


