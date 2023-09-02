import torch.nn as nn
import torch.nn.functional as F


# Create a neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 9, kernel_size = 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(9 * 50 * 12, 2 * 50 * 12)
        self.fc2 = nn.Linear(2 * 50 * 12, 2)
        
    # Specify how data will pass through this model
    def forward(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size = 2, stride = 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size = 2, stride = 2)
        x = x.view(-1, 9 * 50 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)
        return x
