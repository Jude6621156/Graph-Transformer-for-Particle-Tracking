import torch
from torch_geometric.data import Data
import numpy as np

def BuildData(hits, eIndex, eLabels, eFeatures):

    nFeatures = hits[['x', 'y', 'z', 'r', 'phi']].values

    x = torch.tensor(nFeatures, dtype=torch.float32)
    eIndex = torch.tensor(eIndex, dtype=torch.long)
    eAttributes = torch.tensor(np.asarray(eFeatures), dtype=torch.float32)
    if eAttributes.ndim == 1:
        eAttributes = eAttributes.view(-1, 1)
    y = torch.tensor(eLabels, dtype=torch.float32)

    data = Data(x=x, edge_index = eIndex, edge_attr = eAttributes, y=y)

    return data