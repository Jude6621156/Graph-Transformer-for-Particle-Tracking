from src.Models import EdgeClassifier
from src.Pipeline import buildEventData, applyEdgeNorm
from src.Checkpoints import loadCheckpoint

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from collections import Counter

@torch.no_grad()
def predEdge(model, data):
    logits = model(data.x, data.edge_index, data.edge_attr).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()

def getConnectedComps(edge_index, edge_probs, threshold=0.5, min_size=2):
    keep = edge_probs >= threshold
    chosenEdges = edge_index[:, keep]

    G = nx.Graph()
    G.add_edges_from(chosenEdges.T.tolist())

    components = [c for c in nx.connected_components(G) if len(c) >= min_size]
    return components, chosenEdges

def plotRaw(hits, title="Raw Hits"):
    plt.figure(figsize=(8, 8))
    plt.scatter(hits["x"], hits["y"], s=4, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.show()

def predEdges(hits, chosen_edges, title="Predicted Edges"):
    x = hits["x"].to_numpy()
    y = hits["y"].to_numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=3, alpha=0.15, color="gray")

    for i, j in chosen_edges.T:
        plt.plot([x[i], x[j]], [y[i], y[j]], linewidth=0.5, alpha=0.4)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.show()

def plotComponents(hits, components, title = "Connected Components"):
    x = hits["x"].to_numpy()
    y = hits["y"].to_numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=3, alpha=0.15, color = "gray")

    for comp in components:
        comp = list(comp)
        plt.scatter(x[comp], y[comp], s=12, alpha=0.9)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.show()

def compPurity(hits, components):
    particleIds =  hits["particle_id"].to_numpy()
    purityInfo = []

    for i in components:
        i = list(i)
        compPids = particleIds[i]
        compPidsz = compPids[compPids != 0]

        if len(compPidsz) == 0:
            domPid = 0
            domCount = 0
            purity = 0.0
        else:
            counts = Counter(compPidsz)
            domPid, domCount = counts.most_common(1)[0]
            purity = domCount / len(i)

        purityInfo.append({"size": len(i), "dominant particle": int(domPid), "dominant count": domCount, "purity": float(purity)})
    return purityInfo

def synPurity(purityInfo):
    if purityInfo is None or len(purityInfo) == 0:
        print("No components found.")
        return {
        "Number of Components": 0,
        "Average Purity": 0.0,
        "Median Purity": 0.0,
        "Average Track Size": 0.0,
        "Largest Track Size": 0,
        "Purity >= 0.50": 0.0,
        "Purity >= 0.80": 0.0,
        "Purity == 1.00": 0.0
    }

    purities = np.array([p["purity"] for p in purityInfo], dtype=float)
    sizes = np.array([p["size"] for p in purityInfo], dtype=int)

    return {
        "Number of Components": len(purityInfo),
        "Average Purity": float(purities.mean()),
        "Median Purity": float(np.median(purities)),
        "Average Track Size": float(sizes.mean()),
        "Largest Track Size": int(sizes.max()),
        "Purity >= 0.50": float((purities >= 0.50).mean()),
        "Purity >= 0.80": float((purities >= 0.8).mean()),
        "Purity == 1.00": float((purities == 1.00).mean())
    }

def trackConstruct(hits, eIndex, eProbs, thresholds, minSize=2):
    results = []
    for i in thresholds:
        components, chosenEdges = getConnectedComps(eIndex, eProbs, i, minSize)

        purityInfo = compPurity(hits, components)
        synopsis = synPurity(purityInfo)

        results.append({"threshold": float(i),
                        "kept_edges": int(chosenEdges.shape[1]),
                        "components": components,
                        "chosenEdges": chosenEdges,
                        "purity_info": purityInfo,
                        **synopsis
                        })
    return results

def sweepResults(results):
    print("\nThreshold Sweep\n")
    print(
        "\x1B[4m"+(f"{'threshold':>6} | {'edges':>7}   | {'comps':>7}   | {'avgPurity'} | "
        f"{'medPurity':>7} | {'avgSize':>7}   | {'p>=0.8':>7}   | {'p=1.0':>7}   |"+"\x1B[0m")
    )
    for i in results:
        print(
            f"{i['threshold']:6.2f}    |   "
            f"{i['kept_edges']:7d} |   "
            f"{i['Number of Components']:7d} |   "
            f"{i['Average Purity']:7.3f} |   "
            f"{i['Median Purity']:7.3f} |   "
            f"{i['Average Track Size']:7.2f} |   "
            #f"{i['Purity >= 0.50']}"
            f"{i['Purity >= 0.80']:7.3f} |   "
            f"{i['Purity == 1.00']:7.3f} |   "
    )

def bestThreshold(results, min_components=1):
    valid = [j for j in results if j["Number of Components"] >= min_components]
    if not valid:
        return None

    for i in valid:
        i["score"] = (
            i["Average Purity"]
            *np.log1p(i["Average Track Size"])
            * np.log1p(i["Number of Components"])
        )
    return max(valid, key=lambda r: r["score"])

def sweepMetric(results, metric, title):
    thresholds = [i["threshold"] for i in results]
    values = [i[metric] for i in results]

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, values, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root/"Data"/"RawData"/"train_1"
    model_path = project_root/'results'/'best_model.pt'

    default_graph_conf = {"k": 6, "exOutward":True, "maxLayerJump": 2, "maxAbsDphi": 0.4, "maxAbsDzOverDr": 6.0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError('File not found')

    cp = loadCheckpoint(model_path, device)

    eventId = cp.get("val_events", ["event000001000"])[0]
    graph_conf = cp.get("graph_conf", default_graph_conf)
    SampleHits = cp.get("SampleHitsPerEvent", 5000)
    Seed = cp.get("seed", 42)
    thresholds = np.linspace(0.10, 0.80, 9)
    minSize = 2

    d, out = buildEventData(eventId, SampleHits, data_path, graph_conf, device, seed=Seed)
    print(out)

    hits = d["hits"]
    data = d["data"]
    eIndex = d["edge_index"]

    applyEdgeNorm([d], cp["edge_attr_mean"], cp["edge_attr_std"])

    model = EdgeClassifier(data.x.size(1), cp.get("HiddenChannel", 64), data.edge_attr.size(1)).to(device)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    probs = predEdge(model, data)

    print(f"\nEvent: {eventId}")
    print(f"Hits: {len(hits)}")
    print(f"Candidate edges: {data.y.numel()}")
    print(f"Prob min/mean/max: {float(probs.min()):.4f} / {float(probs.mean()):.4f}/{float(probs.max()):.4f}")

    results = trackConstruct(hits, eIndex, probs, thresholds, minSize)

    sweepResults(results)
    best = bestThreshold(results, min_components=5)

    print("\nSelected Threshold")
    print(f"threshold={best['threshold']:.2f}")
    print(f"kept_edges={best['kept_edges']}")
    print(f"components={best['Number of Components']}")
    print(f"avg_purity={best['Average Purity']:.3f}")
    print(f"median_purity={best['Median Purity']:.3f}")
    print(f"avg_size={best['Average Track Size']:.2f}")
    print(f"score = {best['score']:.3f}")

    sweepMetric(results, "Average Purity", "Average Track Purity Against Threshold")
    sweepMetric(results, "Number of Components", "Number of Components Against Threshold")
    sweepMetric(results, "kept_edges", "Kept Edges Against Threshold")

    plotRaw(hits, title=f"Raw Hits | {eventId}")
    predEdges(hits, best['chosenEdges'],title=f"Predicted Edges | threshold={best['threshold']}")
    plotComponents(hits, best['components'], title=f"Connected Components | threshold={best['threshold']}")

if __name__ == "__main__":
    main()




