from src.Data_Loader import load_event
from src.Dataset import BuildData
from src.Models import EdgeClassifier
from src.Graphbuilder import BuildGraphKnn

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

def WeightBalance(labels: torch.Tensor):
    pos = labels.sum().item()
    neg = labels.numel() - pos
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg/pos)

def SplitEdges(eNum, val_frac = 0.2, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.arange(eNum)
    rng.shuffle(ids)
    split = int((1 - val_frac)*eNum)
    return ids[:split], ids[split:]

@torch.no_grad()
def EvalEdges(model, data, eIdx, threshold = 0.5):
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_attr).squeeze(-1)
    prob = torch.sigmoid_(logits)

    yPred = (prob[eIdx].cpu().numpy() >= threshold).astype(int)
    yTrue = data.y[eIdx].cpu().numpy().astype(int)

    prec = precision_score(yTrue, yPred, zero_division=0)
    rec = recall_score(yTrue, yPred, zero_division=0)
    f1 = f1_score(yTrue, yPred, zero_division=0)

    return prec, rec, f1


def main():
    data_path = "../Data/RawData/train_1"
    eventId = "event000001000"
    K = 6
    SampleHits = 5000
    epochs = 25
    LR = 1e-3
    ValFrac = 0.2
    Seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hits, _ =load_event(eventId, data_path)
    SampleNum = min(SampleHits, len(hits))
    hits = hits.sample(SampleNum, random_state=Seed).reset_index(drop=True)

    eIndex, eLabels, eAttribute= BuildGraphKnn(hits, k=K)

    E = eLabels.shape[0]
    data = BuildData(hits, eIndex, eLabels, eAttribute).to(device)

    trainIdx, valIdx = SplitEdges(data.y.numel(), val_frac = ValFrac, seed = Seed)
    trainIdx = torch.tensor(trainIdx, dtype=torch.long, device=device)
    valIdx = torch.tensor(valIdx, dtype=torch.long, device=device)

    NodeDim = data.x.size(1)
    eDim = data.edge_attr.size(1)
    model = EdgeClassifier(InChannel = NodeDim, HiddenChannel=64, eFeaturesSize = eDim).to(device)

    WeightValue = WeightBalance(data.y[trainIdx]).item()
    WeightPos = torch.tensor(min(WeightValue, 50.0), dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=WeightPos)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(1, epochs+1):
        model.train()
        optimiser.zero_grad()

        logits = model(data.x, data.edge_index, data.edge_attr).squeeze(-1)
        loss = criterion(logits[trainIdx], data.y[trainIdx])

        loss.backward()
        optimiser.step()

        if ValFrac > 0:
            prec, rec, f1 = EvalEdges(model, data, valIdx, threshold=0.8)
        else:
            prec, rec, f1 = EvalEdges(model, data, trainIdx, threshold=0.5)
        print(f"Epoch {e:02d} | loss={loss.item():.4f} | val_precision={prec:.3f} val_recall = {rec:.3f} val_f1{f1:.3f}")


if __name__ == "__main__":
    main()