import numpy as np
from sklearn.neighbors import NearestNeighbors

def BuildGraphKnn(hits, k=8,exOutward = False):
    n = len(hits)
    if n == 0:
        return (np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0, 6), dtype=np.float32))

    x = hits['x'].to_numpy(dtype=np.float32)
    y = hits['y'].to_numpy(dtype=np.float32)
    z = hits['z'].to_numpy(dtype=np.float32)
    r = hits['r'].to_numpy(dtype = np.float32)
    phi = hits['phi'].to_numpy(dtype=np.float32)
    layer = hits['layer_id'].to_numpy()
    pid = hits['particle_id'].to_numpy()

    positions = hits[['x', 'y', 'z']].values

    k2 = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors = k2, algorithm = 'ball_tree')
    nn.fit(positions)
    distances, indices = nn.kneighbors(positions)
    eIndex = []
    eLabel = []
    eFeatures = []
    for i in range(n):
        c = 0
        for j in indices[i]:
            if i == j:
                continue

            if abs(layer[j] - layer[i]) > 1:
                continue
            if (r[j] <= r[i]) and exOutward:
                continue

            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]

            dr = r[j] - r[i]
            dphi = phi[j] - phi[i]
            dphi = np.arctan2(np.sin(dphi), np.cos(dphi))

            eucD = np.sqrt(dx**2 + dy**2 + dz**2)

            eIndex.append([i, j])
            eFeatures.append([dx, dy, dz, dr, dphi, eucD])
            eLabel.append(int((pid[i] == pid[j])and (pid[i] != 0)))

            c+=1
            if c >= k:
                break
    if len(eIndex) == 0:
        return (np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0, 6), dtype=np.float32))

    eIndex = np.array(eIndex, dtype=np.int64).T
    eLabel = np.array(eLabel, dtype=np.int64)
    eFeatures = np.array(eFeatures, dtype=np.float32)

    return eIndex, eLabel, eFeatures

