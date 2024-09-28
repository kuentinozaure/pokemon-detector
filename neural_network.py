import torch.nn as nn
import torch.nn.functional as F
import torch

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 input channels (RGB), 12 output channels, 5x5 convolution kernel
        self.conv1 = nn.Conv2d(3, 15, 5)
        
        # 12 input channels, 20 output channels, 5x5 convolution kernel
        self.conv2 = nn.Conv2d(15, 30, 5)

        # Fully connected layer, expects input size of 20 * 8 * 8 (after convolutions and pooling)
        self.fc1 = nn.Linear(30 * 8 * 8, 1024)  # 1280 neurons as input
        self.fc2 = nn.Linear(1024, 800)  # 1024 neurons as input
        self.fc3 = nn.Linear(800, 400)  # 1000 neurons as input
        self.fc4 = nn.Linear(400, 200)  # 500 neurons as input
        self.fc5 = nn.Linear(200, 150)  # 200 neurons as input

        # Max pooling with a kernel size of 5x5 and stride of 5
        self.pool = nn.MaxPool2d(5, 5)

    def forward(self, x):
        # Apply the first convolution, followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: 12 channels of 44x44

        # Apply the second convolution, followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Output: 20 channels of 8x8

        # Flatten the dimensions for the fully connected layer
        x = torch.flatten(x, 1)  # Flatten while keeping the batch dimension, output: 1280

        # Apply the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))  # Output: 1024 neurons

        # Apply the second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))  # Output: 1000 neurons

        # Apply the third fully connected layer with ReLU activation
        x = F.relu(self.fc3(x))  # Output: 500 neurons

        # Apply the fourth fully connected layer with ReLU activation
        x = F.relu(self.fc4(x))  # Output: 200 neurons

        # Apply the fifth fully connected layer, without activation
        x = self.fc5(x)  # Output: 150 neurons (for the final classification)

        return x  # Returns the final predictions
