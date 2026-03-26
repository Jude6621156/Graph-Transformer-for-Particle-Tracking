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
The dataset is not included in the repository and can be found here: https://www.kaggle.com/competitions/trackml-particle-identification/data

The project expects TrackML event data under:

```
Data/RawData/train_1/
```

Each event consists of 4 files:
- *-hits.csv
- *-particles.csv
- *-truth.csv
- *-cells.csv


## Pipeline

### Load Events
Each event group is loaded into a pandas DataFrame containing detector hit information.

### Building a Candidate Graph
Each dataframe is now converted into a graph as follows:
- Nodes represent detector hits
- Edges represent plausible connections between two hits
- Constructed using **k-nearest neighbors (kNN)** in cylindrical space.
- Uses physics informed constraints such as:
  - layer proximity
  - angular difference
  - slope

### Training the GNN Edge Classifier
A GNN is trained to classify whether two hits are connected and belong to the same particle track.

### Reconstructing Track Candidates
- Edges are assigned predictions on how likely it is that they form a track
- A threshold is applied
- Remaining edges are grouped using connected components.
- Each component represents a connected track.

