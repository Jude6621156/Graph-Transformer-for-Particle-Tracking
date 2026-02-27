import numpy as np
from sklearn.neighbors import NearestNeighbors

def BuildGraphKnn(hits, k=8, exOutward = False):
    positions = hits[['x', 'y', 'z']].values

    nn = NearestNeighbors(n_neighbors = k, algorithm = 'ball_tree')
    nn.fit(positions)
    distances, indicies = nn.kneighbors(positions)
    eIndex = []
    eLabel = []
    eFeatures = []
    for i in range(len(indicies)):
        for j in indicies[i]:
            if i == j:
                continue

            layer_i = hits.iloc[i]['layer_id']
            layer_j = hits.iloc[j]['layer_id']
            r_i = hits.iloc[i]['r']
            r_j = hits.iloc[j]['r']

            if abs(layer_j - layer_i) > 1:
                continue
            if r_j <= r_i and exOutward:
                continue


            eIndex.append([i, j])

            Particle = int(hits.iloc[i]['particle_id'] == hits.iloc[j]['particle_id'])
            eLabel.append(Particle)

            dx = hits.iloc[j]['x'] - hits.iloc[i]['x']
            dy = hits.iloc[j]['y'] - hits.iloc[i]['y']
            dz = hits.iloc[j]['z'] - hits.iloc[i]['z']

            dr = hits.iloc[j]['r'] - hits.iloc[i]['r']

            dphi = hits.iloc[j]['phi'] - hits.iloc[i]['phi']
            dphi = np.arctan2(np.sin(dphi), np.cos(dphi))

            eucD = np.sqrt(dx**2 + dy**2 + dz**2)

            eFeatures.append([dx, dy, dz, dr, dphi, eucD])


    eIndex = np.array(eIndex).T
    eLabel = np.array(eLabel)
    eFeatures = np.array(eFeatures)

    return eIndex, eLabel, eFeatures

