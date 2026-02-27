import numpy as np
import pandas as pd
import os

def load_event(eventId, data_path):
    fHits = os.path.join(data_path, f"{eventId}-hits.csv")
    fTruth = os.path.join(data_path, f"{eventId}-truth.csv")
    fParticles = os.path.join(data_path, f"{eventId}-particles.csv")

    hits = pd.read_csv(fHits)
    truth = pd.read_csv(fTruth)
    particles = pd.read_csv(fParticles)

    hits = hits.merge(truth[["hit_id", "particle_id"]], on = "hit_id", how = "left")

    hits['r'] = np.sqrt(hits['x']**2 + hits['y']**2)
    hits['phi'] = np.arctan2(hits['y'], hits['x'])

    return hits, particles