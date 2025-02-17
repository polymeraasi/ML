"""
Convolutional Neural Network (CNN) for Image Classification

This script implements a CNN model in PyTorch designed for image classification.
It follows a typical deep learning pipeline: data loading, model definition, training, and evaluation.

CNN Architecture:
- Conv2D Layer (10 filters, 3x3, stride 2, ReLU)
- MaxPooling (2x2)
- Conv2D Layer (same as first)
- MaxPooling (2x2)
- Fully Connected Layer (2 neurons, Sigmoid)

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


dataset_path = 'C://Users//pinja//PycharmProjects//PR_ML_notebook//GTSRB_subset_2//GTSRB_subset_2'

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
data, labels = zip(*dataset.samples)

train_images, test_images, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
test_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

train_dataset.samples = list(zip(train_images, train_labels))
test_dataset.samples = list(zip(test_images, test_labels))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(10)   
        self.conv2 = nn.Conv2d(10, 10, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(10)
        self.fc = nn.Linear(90, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc(x))
        return x

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

model = CNN()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training the CNN model with train_loader
num_epochs = 20
for epoch in range(num_epochs):
    loss_e = 0
    for train_im, train_lbs in train_loader:
        output = model(train_im)
        loss = loss_function(output.float(), train_lbs)
        loss_e += loss.item() * train_im.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss_e/len(train_loader)}")

# testing the accuracy of the CNN model with test_loader
correct = 0
total = 0
with torch.no_grad():
    for test_im, test_lbl in test_loader:
        output = model(test_im)
        _, predicted = torch.max(output, 1)
        total += test_lbl.size(0)
        correct += (predicted == test_lbl).sum().item()
test_acc = 100 * correct / total
print('\nTest accuracy (%):', test_acc)
