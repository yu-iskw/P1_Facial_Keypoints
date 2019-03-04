## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Input: 1 x 224 x 224
        
        self.conv1 = nn.Conv2d(1, 32, 5) # 32 x 220 x 220
        self.pool1 = nn.MaxPool2d(2, 2) # 32 x 110 x 110
        self.batch_norm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3) # 64 x 108 x 108
        self.pool2 = nn.MaxPool2d(2, 2) # 64 x 54 x 54
        self.batch_norm2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 54 * 54, 256)
        self.fc1_dropout = nn.Dropout(p=0.25)
        self.fc1_batcknorm = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 2*68)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.batch_norm1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.batch_norm2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_dropout(x)
        x = self.fc1_batcknorm(x)
        
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
