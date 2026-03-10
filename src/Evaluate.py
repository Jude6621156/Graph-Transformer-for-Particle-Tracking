import pandas as pd

from src.Data_Loader import load_event
from src.Dataset import BuildData
from src.Models import EdgeClassifier
from src.Graphbuilder import BuildGraphKnn
from src.Training import getEvents, buildEventData

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
import torch
from pathlib import Path
import matplotlib.pyplot as plt


def loadCheckpoint(model_path, node_dim, edge_dim, device):
    cp = torch.load(model_path, map_location=device)

    model = EdgeClassifier(node_dim, 64, edge_dim ).to(device)

    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    return model, cp

def buildEventGraph(eventId, data_path, sampleHits, graph_conf, device, seed=42):
    hits, _ = load_event(eventId, str(data_path))
    n = min(sampleHits, len(hits))
    hits = hits.sample(n, random_state=seed).reset_index(drop=True)

    eIndex, eLabels, eAttributes = BuildGraphKnn(hits, **graph_conf)

    if len(eLabels) == 0:
        raise ValueError(f"No Edges in event {eventId}")

    data = BuildData(hits, eIndex, eLabels, eAttributes).to(device)
    return hits, data, eIndex

@torch.no_grad()
def probsEtLabels(model, data_list):
    model.eval()

    allProbs = []
    allLabels = []

    for d in data_list:
        logits = model(d.x, d.edge_index, d.edge_attr).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = d.y.cpu().numpy()

        allProbs.append(probs)
        allLabels.append(labels)

    return np.concatenate(allProbs), np.concatenate(allLabels)

def compPrecCurve(yTrue, yProbs):
    precision, recall, thresholds = precision_recall_curve(yTrue, yProbs)
    ap = average_precision_score(yTrue, yProbs)

    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + 1e-12)
    bestIdx = np.argmax(f1_scores)

    return{
        "precision_curve": precision,
        "recall_curve": recall,
        "thresholds": thresholds,
        "average_precision": ap,
        "best_threshold": thresholds[bestIdx],
        "best_precision": precision[bestIdx],
        "best_recall": recall[bestIdx],
        "best_f1": f1_scores[bestIdx]
    }

def plotPrecCurve(prec_info, save_path):
    plt.figure(figsize=(8, 8))
    plt.plot(prec_info["recall_curve"], prec_info["precision_curve"], label=f"Prec Curve (AP={prec_info['average_precision']:.4f})")
    plt.scatter(prec_info["best_recall"], prec_info["best_precision"], label=f"Best F1 @ {prec_info['best_threshold']:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=200)
    plt.show()


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root /"Data"/"RawData"/"train_1"
    model_path = project_root/"results"/"best_model.pt"
    plots_dir = project_root/"results"/"plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    NumValEvents = 20
    SampleHitsPerEvent = 3000
    Seed = 42

    default_graph_conf = {"k": 6, "exOutward": True, "maxAbsDphi": 0.4, "maxAbsDzOverDr": 6.0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cp = torch.load(model_path, map_location=device)
    graph_conf = cp.get("graph_conf", default_graph_conf)

    ValEvents = cp.get("val_events", [])
    if len(ValEvents) == 0:
        _, ValEvents = getEvents(data_path=data_path, n_train=0, n_val=NumValEvents, seed=Seed)

    valData = []
    for i, j in enumerate(ValEvents):
        d = buildEventData(j, data_path, SampleHitsPerEvent, graph_conf, device, seed=Seed)
        if d is not None:
            valData.append(d)

    if len(valData) == 0:
        raise ValueError("No events")



    model = EdgeClassifier(valData[0].x.size(1), 64, valData[0].edge_attr.size(1)).to(device)

    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    probs, labels = probsEtLabels(model, valData)
    prec_info = compPrecCurve(labels, probs)

    print(f"Average Precision: {prec_info['average_precision']:.4f}")
    print(f"Best Threshold: {prec_info['best_threshold']:.4f}")
    print(f"Best Precision: {prec_info['best_precision']:.4f}")
    print(f"Best Recall: {prec_info['best_recall']:.4f}")
    print(f"Best f1: {prec_info['best_f1']:.4f}")

    plotPrecCurve(prec_info, plots_dir / "precision_recall_curve.png")

if __name__ == '__main__':
    main()
