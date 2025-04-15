import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Standard normalization
])

# Load CIFAR-10 training and test sets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Define CNN-LSTM Model
class CNNLSTM(nn.Module):
    def __init__(self, hidden_neurons=100):
        super(CNNLSTM, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (batch, 32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (batch, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch, 64, 8, 8)
        )

        # Adjust the LSTM input size to 64 * 8 (8x8 spatial dimensions)
        self.lstm = nn.LSTM(input_size=64 * 8, hidden_size=hidden_neurons, num_layers=1, batch_first=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_neurons * 8, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)  # Output layer (10 classes)
        )

    def forward(self, x):
        x = self.cnn(x)  # (batch, 64, 8, 8)
        x = x.permute(0, 2, 3, 1)  # (batch, 8, 8, 64)
        x = x.reshape(x.size(0), 8, 64 * 8)  # Reshape to (batch, 8, 512)

        lstm_out, _ = self.lstm(x)  # (batch, 8, hidden_neurons)
        lstm_out = lstm_out.contiguous().reshape(x.size(0), -1)  # Flatten the LSTM output

        out = self.fc(lstm_out)
        return out


# Train and evaluate the model
def train_and_evaluate(hidden_neurons, optimizer_type, num_epochs):
    model = CNNLSTM(hidden_neurons).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Evaluate different configurations and store the results
results = []
neurons_list = [100, 200]
epochs_list = [50, 100]
optimizers = ['adam', 'sgd']

for h in neurons_list:
    for e in epochs_list:
        for opt in optimizers:
            acc = train_and_evaluate(h, opt, e)
            results.append({
                'Neurons': h,
                'Epochs': e,
                'Optimizer': opt.upper(),
                'Accuracy': acc
            })

# Create a DataFrame to show results
df_results = pd.DataFrame(results)
print(df_results)
