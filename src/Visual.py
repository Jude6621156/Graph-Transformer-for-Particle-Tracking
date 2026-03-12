from src.Data_Loader import load_event
from src.Dataset import BuildData
from src.Models import EdgeClassifier
from src.Graphbuilder import BuildGraphKnn

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from collections import Counter

def loadModel(model_path, node_dim, edge_dim, device):
    model = EdgeClassifier(InChannel=node_dim, HiddenChannel=64, eFeaturesSize=edge_dim).to(device)
    cp = torch.load(model_path, map_location=device)

    if isinstance(cp, dict) and "model_state_dict" in cp:
        state = cp["model_state_dict"]
    elif isinstance(cp, dict) and "model state dict" in cp:
        state = cp["model state dict"]
    else:
        state = cp

    model.load_state_dict(state)
    model.eval()
    bestThreshold = 0.5
    if isinstance(cp, dict):
        bestThreshold = cp.get("best threshold", cp.get("best_threshold", 0.5))

    return model, bestThreshold, cp

def buildEventGraph(event_id, data_path, sample_hits, graph_conf, device, seed=42):
    hits, _ = load_event(event_id, str(data_path))
    n = min(sample_hits, len(hits))
    hits = hits.sample(n, random_state=seed).reset_index(drop=True)

    eIndex, eLabels, eAttributes = BuildGraphKnn(hits, **graph_conf)
    data = BuildData(hits, eIndex, eLabels, eAttributes).to(device)

    return hits, data, eIndex

@torch.no_grad()
def predEdge(model, data):
    logits = model(data.x, data.edge_index,data.edge_attr).squeeze(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def getConnectedComps(edge_index, edge_probs, threshold=0.9, min_size=3):
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

def plotCompnents(hits, components, title = "Connected Components"):
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
            dommyPid = 0
            dommyCount = 0
            purity = 0.0
        else:
            counts = Counter(compPidsz)
            dommyPid, dommyCount = counts.most_common(1)[0]
            purity = dommyCount / len(i)

        purityInfo.append({"size": len(i), "dominant particle": int(dommyPid), "dominant count": dommyCount, "purity": float(purity)})
    return purityInfo

def synPurity(purityInfo):
    if purityInfo is None or len(purityInfo) == 0:
        print("'Tis emppty my friends.")
        return

    purities = np.array([p["purity"] for p in purityInfo], dtype=float)
    sizes = np.array([p["size"] for p in purityInfo], dtype=int)

    print("\n Track Purity:")
    print(f"Number of components: {len(purityInfo)}")
    print(f"Average Purity: {purities.mean():.3f}")
    print(f"Median Purity: {np.median(purities):.3f}")
    print(f"Average Track Size: {sizes.mean():.2f}")
    print(f"Largest Track Size: {sizes.max()}")
    print(f"Purity >= 0.50: {(purities >= 0.50).mean():.3f}")
    print(f"Purity >= 0.80: {(purities >= 0.8).mean():.3f}")
    print(f"Purity == 1.00: {(purities == 1.00).mean():.3f}")

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root/"Data"/"RawData"/"train_1"
    model_path = project_root/'results'/'best_model.pt'
    eventId = "event000001000"

    SampleHits = 10000
    Seed = 42
    threshold = 0.25
    minSize = 2

    default_graph_conf = {"k": 6, "exOutward":True, "maxLayerJump": 2, "maxAbsDphi": 0.4, "maxAbsDzOverDr": 6.0}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError('File not found')

    cp = torch.load(model_path, map_location=device)
    graph_conf = cp.get("graph_conf", default_graph_conf)
   # graph_conf = {"k": 8, "exOutward":True, "maxLayerJump": 2, "maxAbsDphi": 0.4, "maxAbsDzOverDr": 6.0}
    hits, data, eIndex = buildEventGraph(eventId, data_path, SampleHits, graph_conf, device, seed=Seed)

    model, cp_threshold, cp_meta= loadModel(model_path, data.x.size(1), data.edge_attr.size(1), device=device)
    threshold = 0.3

    probs = predEdge(model, data)

    components, chosenEdges = getConnectedComps(eIndex, probs, threshold, minSize)

    purityInfo = compPurity(hits, components)
    synPurity(purityInfo)

    print(f"Event: {eventId}")
    print(f"Hits: {len(hits)}")
    print(f"Edges: {data.y.numel()}")
    print(f"Threshold: {threshold}")
    print(f"Components Found: {len(components)}")

    plotRaw(hits, title=f"Raw Hits | {eventId}")
    predEdges(hits, chosenEdges,title=f"Predicted Edges | threshold={threshold}")
    plotCompnents(hits, components, title=f"Connected Components | threshold={threshold}")

if __name__ == "__main__":
    main()




