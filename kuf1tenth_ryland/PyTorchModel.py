import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        
        # First layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=268, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second layer
        self.conv2 = nn.Conv1d(in_channels=268, out_channels=66, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third layer
        self.conv3 = nn.Conv1d(in_channels=66, out_channels=64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Flatten
        self.flatten = nn.Flatten()
        # Dense layers
        self.fc1 = nn.Linear(960, 100) 
        self.fc2 = nn.Linear(100, 80)
        self.fc3 = nn.Linear(80, 50)
        self.fc4 = nn.Linear(50, 2)
    


    def forward(self, x):
        # First layer
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Second layer
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Third layer
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten
        x = self.flatten(x)  # Flatten all dimensions except the batch dimension
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Output layer with tanh activation
        
        return x