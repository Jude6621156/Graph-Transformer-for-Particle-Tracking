import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn

class EdgeClassifier(nn.Module):
    def __init__(self, InChannel, HiddenChannel, eFeaturesSize):
        super().__init__()

        self.conv1 = GCNConv(InChannel, HiddenChannel)
        self.conv2 = GCNConv(HiddenChannel, HiddenChannel)

        self.eMLP = nn.Sequential(nn.Linear(2*HiddenChannel + eFeaturesSize, HiddenChannel), nn.ReLU(), nn.Linear(HiddenChannel, 1))

    def forward(self, x, eIndex, eAttributes):
        x = self.conv1(x, eIndex)
        x = torch.relu(x)
        x = self.conv2(x, eIndex)

        row, col = eIndex
        eFeatures = torch.cat([x[row], x[col], eAttributes], dim=1)

        return self.eMLP(eFeatures)