# Particle Tracking With Graph Neural Networks

This project explores the use of **Graph Neural Networks (GNNs)** to reconstruct particle trajectories from detector hits.


## Objective
Particle detectors record thousands of particle hits, but they do not indicate which hits correspond to which particles.

This project aims to:
- Construct a graph of candidate connections between hits
- Train a GNN to classify whether two hits belong to the same particle
- Accurately reconstruct tracks using connected components

---

## Dataset
The dataset is not included in the repository and can be found here: 
https://www.kaggle.com/competitions/trackml-particle-identification/data

The project expects TrackML event data under:

```
Data/RawData/train_1/
```

Each event consists of 4 files:
- *-hits.csv
- *-particles.csv
- *-truth.csv
- *-cells.csv

---

## Pipeline

### Load Events
Each event is loaded into a pandas DataFrame containing detector hit information.

### Building a Candidate Graph
Each event is converted into a graph where:
- Nodes represent detector hits
- Edges represent plausible connections between hits

Edges are constructed using **k-nearest neighbors**(kNN) in cylindrical coordinate space, with additional physics informed constraints such as:
  - layer proximity
  - angular difference
  - gradient (Δz/ Δr) 
This reduces search space while keeping physically plausible connections. 

### Training the GNN Edge Classifier
A GNN is trained to classify whether two hits are connected and belong to the same particle track.

Node representations are learned by message passing.
Edge classification is performed using a multilayer perceptron on:
- node embeddings
- engineered geometric edge features

---

### Reconstructing Track Candidates
- Edges are each assigned a probability on how likely it is that they form a track
- A threshold is applied to select likely connections
- Remaining edges are grouped using **connected components**
- Each component represents a reconstructed track

---

## Results
In the current configuration, validation edge classification F1 exceeds 0.9 after roughly 60-70 epochs.

As for reconstructed tracks there is a trade-off between:
- **Track Purity** (Accuracy of reconstructed tracks)
- **Track Continuity** (length and completeness of tracks)
 
Lower thresholds produce longer but noisier tracks, while higher thresholds produce shorter but cleaner track fragments, as can be seen below:

![img.png](images%2Fimg.png)

### Precision-Recall Graph
Shows precision measured against recall

![Figure_7.png](images%2FFigure_72.png)

### Track Purity
Track purity at different thresholds

![Figure_8.png](images%2FFigure_8.png)

