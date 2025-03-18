# models/dgcnn.py

import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv

class DGCNN(nn.Module):
    def __init__(self, num_classes):
        super(DGCNN, self).__init__()
        self.conv1 = DynamicEdgeConv(nn=nn.Sequential(nn.Linear(6, 64), nn.ReLU()), k=20)
        self.conv2 = DynamicEdgeConv(nn=nn.Sequential(nn.Linear(128, 128), nn.ReLU()), k=20)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x