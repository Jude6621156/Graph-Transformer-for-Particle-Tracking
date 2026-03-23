from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score
import torch

from src.Graphbuilder import BuildGraphKnn
from src.Dataset import BuildData
from src.Data_Loader import load_event

def getEventId(data_path):
    hit_files = sorted(Path(data_path).glob("event*-hits.csv"))
    return [p.stem.replace("-hits", "") for p in hit_files]

def getEvents(data_path, n_train, n_val, seed=42):
    eventIds = getEventId(data_path)
    n = n_train + n_val
    if len(eventIds) < n:
        raise ValueError(f"Requested {n} events, only {len(eventIds)} available")

    rng = np.random.default_rng((seed))
    perm = rng.permutation(len(eventIds))
    events = [eventIds[i] for i in perm[:n]]
    trainEvents = events[:n_train]
    valEvents = events[n_train:n_train+n_val]
    return trainEvents, valEvents

def buildEventData(event_id, sample_hits, data_path, graph_conf, device, seed):
    hits, _ = load_event(event_id, str(data_path))
    n = min(sample_hits, len(hits))
    hits = hits.sample(n, random_state=seed).reset_index(drop=True)

    eIndex, eLabels, eAttributes = BuildGraphKnn(hits, **graph_conf)
    if len(eLabels) == 0:
        return None, f"Skipping {event_id}: No edges post graph build"

    pos = int(np.sum(eLabels))
    if pos == 0:
        return None, f"Skipping {event_id}: No positive edges"

    pos_ratio = float(np.mean(eLabels))
    msg = f"{event_id}: edges={len(eLabels)} pos={pos} pos_ratio={pos_ratio:.6f}"

    data = BuildData(hits, eIndex, eLabels, eAttributes).to(device)
    return {"hits": hits, "data": data, "edge_index": eIndex}, msg

def fitEdgeNorm(train_items):
    edgesAttr = torch.cat([i["data"].edge_attr.detach().cpu() for i in train_items])
    mean = edgesAttr.mean(dim=0)
    std = edgesAttr.std(dim=0).clamp_min(1e-6)
    return mean, std

def applyEdgeNorm(items, mean, std):
    for i in items:
        data = i["data"]
        data.edge_attr = (data.edge_attr - mean.to(data.edge_attr.device)) / std.to(data.edge_attr.device)

@torch.no_grad()
def EvalEdges(model, data, threshold = 0.5):
    model.eval()

    yTrues = []
    yPreds = []

    for j in data:
        data = j["data"]
        logits = model(data.x, data.edge_index, data.edge_attr).squeeze(-1)
        prob = torch.sigmoid_(logits)

        yPred = (prob>=threshold).cpu().numpy().astype(int)
        yTrue = data.y.cpu().numpy().astype(int)

        yTrues.append(yTrue)
        yPreds.append(yPred)

    yTrues = np.concatenate(yTrues)
    yPreds = np.concatenate(yPreds)

    prec = precision_score(yTrues, yPreds, zero_division=0)
    rec = recall_score(yTrues, yPreds, zero_division=0)
    f1 = f1_score(yTrues, yPreds, zero_division=0)

    return prec, rec, f1

def sweepThresholds(model, data, thresholds):
    best = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
    for t in thresholds:
        prec, rec, f1 = EvalEdges(model, data, threshold=t)
        if f1 > best['f1']:
            best = {"f1": f1, "precision": prec, "recall": rec, "threshold": t}
    return best