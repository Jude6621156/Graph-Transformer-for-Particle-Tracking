import torch
from torch_geometric.data import Data

def BuildData(hits, eIndex, eLabels, eFeatures):

    nFeatures = hits[['x', 'y', 'z', 'r', 'phi']].values

    x = torch.tensor(nFeatures)
    eIndex = torch.tensor(eIndex)
    eAttributes = torch.tensor(eFeatures)
    y = torch.tensor(eLabels)

    data = Data(x=x, edge_index = eIndex, edge_attr = eAttributes, y=y)

    return data