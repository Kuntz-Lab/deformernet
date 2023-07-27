from __future__ import print_function
# import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 3)

        # self.bn1 = nn.GroupNorm(1, 1)
        # self.bn2 = nn.GroupNorm(1, 1)
        # self.bn3 = nn.GroupNorm(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        # x = self.bn3(x)
        x = F.relu(x)
        
        output = self.fc4(x)
   
        return output

def train(model, device, optimizer, epoch, data, target):
    model.train()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    
    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, loss.item()))