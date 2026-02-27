from src.Data_Loader import load_event
import matplotlib.pyplot as plt
from src.Graphbuilder import BuildGraphKnn


data_path = "../Data/RawData/train_1"
eventId = "event000001000"

hits, particles = load_event(eventId, data_path)

hits = hits.sample(5000).reset_index(drop=True)

eIndex, eLabel = BuildGraphKnn(hits, k = 6)

print("Edges: ", eIndex.shape)
print("Positive Edges: ", eLabel.sum())
print("Negative Edges: ", len(eLabel) - eLabel.sum())


