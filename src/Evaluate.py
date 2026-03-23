from src.Models import EdgeClassifier
from src.Pipeline import buildEventData, getEvents, applyEdgeNorm
from src.Checkpoints import loadCheckpoint

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
from pathlib import Path
import matplotlib.pyplot as plt

@torch.no_grad()
def probsEtLabels(model, data_list):
    model.eval()

    allProbs = []
    allLabels = []

    for d in data_list:
        data = d["data"]
        logits = model(data.x, data.edge_index, data.edge_attr).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = data.y.cpu().numpy()

        allProbs.append(probs)
        allLabels.append(labels)

    return np.concatenate(allProbs), np.concatenate(allLabels)

def compPrecCurve(yTrue, yProbs):
    precision, recall, thresholds = precision_recall_curve(yTrue, yProbs)
    ap = average_precision_score(yTrue, yProbs)

    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
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

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root /"Data"/"RawData"/"train_1"
    model_path = project_root/"results"/"best_model.pt"
    plots_dir = project_root/"results"/"plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cp = loadCheckpoint(model_path, device)
    graph_conf = cp["graph_conf"]

    ValEvents = cp.get("val_events", [])
    if len(ValEvents) == 0:
        _, ValEvents = getEvents(data_path=data_path, n_train=0, n_val=20, seed=cp.get("seed", 42))

    valData = []
    for i, j in enumerate(ValEvents):
        d, out = buildEventData(j, cp.get("SampleHitsPerEvent", 5000), data_path, graph_conf, device, seed=cp.get("seed", 42)+i)
        print(out)
        if d is not None:
            valData.append(d)

    if len(valData) == 0:
        raise ValueError("No events")

    applyEdgeNorm(valData, cp['edge_attr_mean'], cp['edge_attr_std'])

    model = EdgeClassifier(valData[0]["data"].x.size(1), cp.get("HiddenChannel", 64), valData[0]["data"].edge_attr.size(1)).to(device)

    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    probs, labels = probsEtLabels(model, valData)
    prec_info = compPrecCurve(labels, probs)

    print(f"Average Precision: {prec_info['average_precision']:.4f}")
    print(f"Best Threshold: {prec_info['best_threshold']:.4f}")
    print(f"Best Precision: {prec_info['best_precision']:.4f}")
    print(f"Best Recall: {prec_info['best_recall']:.4f}")
    print(f"Best f1: {prec_info['best_f1']:.4f}")

    plt.figure(figsize=(8,8))
    plt.plot(
        prec_info["recall_curve"],
        prec_info["precision_curve"],
        label=f"Precision Curve (AP={prec_info['average_precision']:.4f})"
    )
    plt.scatter(
        prec_info["best_recall"],
        prec_info["best_precision"],
        label=f"Best F1 @ {prec_info['best_threshold']:.3f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(plots_dir/ "precision_recall_curve.png", dpi=200)
    plt.show()
if __name__ == '__main__':
    main()
