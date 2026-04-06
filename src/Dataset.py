import torch
from torch_geometric.data import Data
import numpy as np

# Transforms raw data into graph
def BuildData(hits, eIndex, eLabels, eFeatures):

    # Node features are cartesian plus cylindrical coordinates
    nFeatures = hits[['x', 'y', 'z', 'r', 'phi']].values

    x = torch.tensor(nFeatures, dtype=torch.float32)
    eIndex = torch.tensor(eIndex, dtype=torch.long)
    eAttributes = torch.tensor(np.asarray(eFeatures), dtype=torch.float32)
    # Ensure edge attributes are always 2d as pytorch expects
    if eAttributes.ndim == 1:
        eAttributes = eAttributes.view(-1, 1)
    y = torch.tensor(eLabels, dtype=torch.float32)

    data = Data(x=x, edge_index = eIndex, edge_attr = eAttributes, y=y)

    return data