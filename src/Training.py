from src.Data_Loader import load_event
from src.Dataset import BuildData
from src.Models import EdgeClassifier
from src.Graphbuilder import BuildGraphKnn

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from pathlib import Path

def getEventId(data_path):
    hit_files = sorted(Path(data_path).glob("event*-hits.csv"))
    return [p.stem.replace("-hits", "") for p in hit_files]

def getEvents(data_path, n_train, n_val, seed=42):
    eventIds = getEventId(data_path)
    need = n_train + n_val
    if len(eventIds) < need:
        raise ValueError(f"Requested {need} events, found only {len(eventIds)} in {data_path}")

    r = np.random.default_rng(seed)
    perm = r.permutation(len(eventIds))
    chosen = [eventIds[i] for i in perm[:need]]

    train_events = chosen[:n_train]
    val_events = chosen[n_train:n_train+n_val]
    return train_events, val_events

def buildEventData(event_id, data_path, sample_hits, graph_conf, device, seed):
    hits, _ = load_event(event_id, str(data_path))
    n = min(sample_hits, len(hits))
    hits = hits.sample(n, random_state=seed).reset_index(drop=True)

    eIndex, eLabels, eAttributes = BuildGraphKnn(hits, **graph_conf)

    if len(eLabels) == 0:
        print(f"Skipping {event_id}: no edges after graph build")
        return None

    pos = int(np.sum(eLabels))
    if pos == 0:
        print(f"Skipping event {event_id}, no poisitve edges")
        return None

    pos_ratio = float(np.mean(eLabels))
    print(f"{event_id}: edges={len(eLabels)} pos={pos} pos_ratio={pos_ratio}")

    return BuildData(hits, eIndex, eLabels, eAttributes).to(device)

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
def EvalEdges(model, data, threshold = 0.5):
    model.eval()

    yTrues = []
    yPreds = []

    for j in data:
        logits = model(j.x, j.edge_index, j.edge_attr).squeeze(-1)
        prob = torch.sigmoid_(logits)

        yPred = (prob>=threshold).cpu().numpy().astype(int)
        yTrue = j.y.cpu().numpy().astype(int)

        yTrues.append(yTrue)
        yPreds.append(yPred)

    yTrues = np.concatenate(yTrues)
    yPreds = np.concatenate(yPreds)

    prec = precision_score(yTrues, yPreds, zero_division=0)
    rec = recall_score(yTrues, yPreds, zero_division=0)
    f1 = f1_score(yTrues, yPreds, zero_division=0)

    return prec, rec, f1

def NegativeSampling( y, neg_ratio=10, seed=42,device='cpu', logits = None, HardFrac = 0.5):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    PosIdx = torch.where(y==1)[0].cpu()
    NegIdx = torch.where(y==0)[0].cpu()

    PosNum = PosIdx.numel()
    NegNum = NegIdx.numel()

    if PosNum == 0:
        raise ValueError("No positive edges.")

    NegSample = min(neg_ratio * PosNum, NegNum)
    HardNum = min(int(HardFrac * NegSample), NegNum) if logits is not None else 0

    HardNeg = torch.empty(0, dtype=torch.long)
    if HardNum > 0:
        NegScores = torch.sigmoid(logits.detach())[NegIdx.to(logits.device)].cpu()
        HardLocal = torch.topk(NegScores, k=HardNum, largest=True).indices
        HardNeg = NegIdx[HardLocal]

    remaining = NegSample - HardNeg.numel()
    if remaining > 0:
        if HardNeg.numel() > 0:
            used = torch.zeros(NegNum, dtype=torch.bool)
            HardSet = set(HardNeg.tolist())
            for i, idx in enumerate(NegIdx.tolist()):
                if idx in HardSet:
                    used[i] = True
            pool = NegIdx[~used]
        else:
            pool = NegIdx

        perm = torch.randperm(pool.numel(), generator=g)[:remaining]
        RandNeg = pool[perm]
    else:
        RandNeg = torch.empty(0, dtype=torch.long)

    trainIdx = torch.cat([PosIdx, HardNeg, RandNeg])
    trainIdx = trainIdx[torch.randperm(trainIdx.numel(), generator=g)]

    return trainIdx.to(device)


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root/"Data"/"RawData"/"train_1"
    model_path = project_root/"results"/"best_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    NumTrainEvents = 50
    NumValEvents = 50
    SampleHitsPerEvent = 3000

    K = 6
    epochs = 50
    LR = 5e-4
    HardFrac = 0.1
    Thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
    NegRatio = 5
    Seed = 42
    maxAbsDphi = 0.4
    maxAbsDzOverDr = 6.0

    graph_conf = {"k": K, "exOutward":True, "maxLayerJump": 1, "maxAbsDphi": 0.4, "maxAbsDzOverDr": 6.0}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TrainEvents, ValEvents = getEvents(data_path=data_path, n_train=NumTrainEvents, n_val=NumValEvents, seed=Seed)

    print("Train Events:", TrainEvents)
    print("Val Events:", ValEvents)

    trainData = []
    for i, j in enumerate(TrainEvents):
        d = buildEventData(j, data_path, SampleHitsPerEvent, graph_conf, device, seed=Seed+i)
        if d is not None:
            trainData.append(d)

    valData = []
    for i, j in enumerate(ValEvents):
        d = buildEventData(j, data_path, SampleHitsPerEvent, graph_conf, device, seed=Seed + 100 + i)
        if d is not None:
            valData.append(d)

    if len(trainData) == 0:
        raise ValueError("No usable train events.")
    if len(valData) == 0:
        raise ValueError("No usable validation events.")

    NodeDim = trainData[0].x.size(1)
    eDim = trainData[0].edge_attr.size(1)
    model = EdgeClassifier(InChannel = NodeDim, HiddenChannel=64, eFeaturesSize = eDim).to(device)

    #posTotal = sum(int((d.y==1).sum().item())for d in trainData)
    #negTotal = sum(int((d.y==0).sum().item())for d in trainData)
    #pw = max(1.0, negTotal/max(1, posTotal))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    best = {"epoch": 0, "f1": -1.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}

    for e in range(1, epochs+1):
        model.train()
        EpochLoss = 0.0
        for i, j in enumerate(trainData):
            optimiser.zero_grad()

            logits = model(j.x, j.edge_index, j.edge_attr).squeeze(-1)

            trainIdx = NegativeSampling(j.y, neg_ratio=NegRatio, seed=Seed + (e*1000)+i, device=device, logits=logits if HardFrac>0 else None,
                                        HardFrac=HardFrac)

            loss = criterion(logits[trainIdx], j.y[trainIdx])

            loss.backward()
            optimiser.step()

            EpochLoss +=loss.item()

        bestEpoch = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
        for t in Thresholds:
            prec, rec, f1 = EvalEdges(model, valData, threshold=t)
            if f1>bestEpoch['f1']:
                bestEpoch = {"f1": f1, "precision": prec, "recall": rec, "threshold": t}

        if bestEpoch['f1']>best['f1']:
            best = {'epoch': e, **bestEpoch}

            torch.save({"seed": Seed, "train_events": TrainEvents, "val_events": ValEvents,"epoch": best['epoch'], "model_state_dict": model.state_dict(), "precision": best['precision'], "recall": best['recall'], "f1": best['f1'], "best_threshold": best['threshold'], "graph_conf": graph_conf}, model_path)
            print(f"Saved model: {model_path}")

        print(f"Epoch {e:02d} | loss={EpochLoss/len(trainData):.4f} | "f"val_precision={bestEpoch['precision']:.3f} "f"val_recall = {bestEpoch[f'recall']:.3f}" f"val_f1={bestEpoch['f1']:.3f}" f"| best_t={bestEpoch['threshold']:.2f}")
    print(f"Best epoch={best['epoch']} | precision={best['precision']:.3f}"
          f"recall={best['recall']:.3f} f1={best['f1']:.3f} threshold={best['threshold']:.2f}")


if __name__ == "__main__":
    main()