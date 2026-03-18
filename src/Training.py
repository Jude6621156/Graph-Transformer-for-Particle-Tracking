from src.Data_Loader import load_event
from src.Dataset import BuildData
from src.Models import EdgeClassifier
from src.Graphbuilder import BuildGraphKnn
from src.Pipeline import getEvents, buildEventData, fitEdgeNorm, applyEdgeNorm, sweepThresholds
from src.Models import EdgeClassifier
from src.Checkpoints import saveCheckpoint

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from pathlib import Path

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

    conf = {
    "NumTrainEvents": 100,
    "NumValEvents": 100,
    "SampleHitsPerEvent": 5000,
    "epochs": 100,
    "LR": 5e-4,
    "HardFrac": 0.3,
    "Thresholds": np.linspace(0.1, 0.9, 17),
    "NegRatio": 5,
    "Seed": 42,
    "graph_conf": {
        "k": 16,
        "exOutward": True,
        "maxLayerJump": 2,
        "maxAbsDphi": 0.6,
        "maxAbsDzOverDr": 8.0,
    }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TrainEvents, ValEvents = getEvents(data_path=data_path,
                                       n_train=conf["NumTrainEvents"],
                                       n_val=conf["NumValEvents"],
                                       seed=conf["Seed"])

    print("Train Events:", TrainEvents)
    print("Val Events:", ValEvents)

    trainData = []
    for i, j in enumerate(TrainEvents):
        d, out = buildEventData(j, conf["SampleHitsPerEvent"], data_path, conf["graph_conf"], device, seed=conf["Seed"]+i)
        print(out)
        if d is not None:
            trainData.append(d)

    valData = []
    for i, j in enumerate(ValEvents):
        d, out = buildEventData(j, conf["SampleHitsPerEvent"], data_path, conf["graph_conf"], device, seed=conf["Seed"] + 100 + i)
        print(out)
        if d is not None:
            valData.append(d)

    if len(trainData) == 0:
        raise ValueError("No usable train events.")
    if len(valData) == 0:
        raise ValueError("No usable validation events.")

    edge_attr_mean, edge_attr_std = fitEdgeNorm(trainData)
    applyEdgeNorm(trainData, edge_attr_mean, edge_attr_std)
    applyEdgeNorm(valData, edge_attr_mean, edge_attr_std)

    NodeDim = trainData[0]["data"].x.size(1)
    eDim = trainData[0]["data"].edge_attr.size(1)
    model = EdgeClassifier(InChannel = NodeDim, HiddenChannel=64, eFeaturesSize = eDim).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=conf["LR"])

    best = {"epoch": 0, "f1": -1.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}

    for e in range(1, conf["epochs"]+1):
        model.train()
        EpochLoss = 0.0
        for i, j in enumerate(trainData):
            optimiser.zero_grad()

            data = j['data']
            logits = model(data.x, data.edge_index, data.edge_attr).squeeze(-1)

            trainIdx = NegativeSampling(data.y, neg_ratio=conf["NegRatio"], seed=conf["Seed"] + (e*1000)+i, device=device, logits=logits if conf["HardFrac"]>0 else None,
                                        HardFrac=conf["HardFrac"])

            loss = criterion(logits[trainIdx], data.y[trainIdx])

            loss.backward()
            optimiser.step()

            EpochLoss +=loss.item()

        bestEpoch = sweepThresholds(model, valData, conf["Thresholds"])
        if best["f1"]<bestEpoch['f1']:
            best = {"epoch": e, **bestEpoch}
            saveCheckpoint(
                model_path,
                model,
                best,
                conf["graph_conf"],
                TrainEvents,
                ValEvents,
                edge_attr_mean,
                edge_attr_std
            )
            print(f"Saved model: {model_path}")

        print(
             f"Epoch {e:02d} | loss={EpochLoss/len(trainData):.4f} |"
             f"val_precision={bestEpoch['precision']:.3f} "
             f"val_recall = {bestEpoch[f'recall']:.3f}" 
             f"val_f1={bestEpoch['f1']:.3f}" 
             f"| best_t={bestEpoch['threshold']:.2f}")
    print(f"Best epoch={best['epoch']} | precision={best['precision']:.3f}"
           f"recall={best['recall']:.3f} f1={best['f1']:.3f} threshold={best['threshold']:.2f}")


if __name__ == "__main__":
    main()