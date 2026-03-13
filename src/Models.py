import torch
from torch_geometric.nn import GATv2Conv
import torch.nn as nn

class EdgeClassifier(nn.Module):
    def __init__(self, InChannel, HiddenChannel, eFeaturesSize):
        super().__init__()

        self.nodeEncoder = nn.Sequential(
            nn.Linear(InChannel, HiddenChannel),
            nn.ReLU(),
            nn.Linear(HiddenChannel, HiddenChannel)
        )

        self.conv1 = GATv2Conv(HiddenChannel, HiddenChannel, heads=2, concat=False)
        self.conv2 = GATv2Conv(HiddenChannel, HiddenChannel, heads=2, concat=False)
        self.conv3 = GATv2Conv(HiddenChannel, HiddenChannel, heads=2, concat=False)

        self.eMLP = nn.Sequential(
            nn.Linear(2 * HiddenChannel +eFeaturesSize, HiddenChannel),
            nn.ReLU(),
            nn.Linear(HiddenChannel, HiddenChannel // 2),
            nn.ReLU(),
            nn.Linear(HiddenChannel // 2, 1)
        )

    def forward(self, x, eIndex, eAttributes):
        x0 = self.nodeEncoder(x)

        x1 = self.conv1(x0, eIndex)
        x1 = torch.relu(x1)

        x2 = self.conv2(x1, eIndex)
        x2 = torch.relu(x2)

        x3 = self.conv3(x2, eIndex)
        x3 = torch.relu(x3)

        x = x0 + x1 + x2 + x3

        row, col = eIndex
        eFeatures = torch.cat([x[row], x[col], eAttributes], dim=1)

        return self.eMLP(eFeatures)