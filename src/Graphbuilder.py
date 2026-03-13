import numpy as np
from sklearn.neighbors import NearestNeighbors

def BuildGraphKnn(hits, k=12, exOutward = False, maxLayerJump = 2, maxAbsDphi=None, maxAbsDzOverDr=None, physSPace=True, phiScale=50.0, zScale=0.02):
    n = len(hits)
    if n == 0:
        return (np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0, 11), dtype=np.float32))

    x = hits['x'].to_numpy(dtype=np.float32)
    y = hits['y'].to_numpy(dtype=np.float32)
    z = hits['z'].to_numpy(dtype=np.float32)
    r = hits['r'].to_numpy(dtype = np.float32)
    phi = hits['phi'].to_numpy(dtype=np.float32)
    layer = hits['layer_id'].to_numpy()
    pid = hits['particle_id'].to_numpy()

    if physSPace:
        knnSpace = np.stack([r, np.sin(phi)*phiScale, np.cos(phi)*phiScale,z*zScale], axis=1)
    else:
        knnSpace = np.stack([x, y, z], axis=1)

    kQuery = min(3 * k + 1, n)
    nn = NearestNeighbors(n_neighbors = kQuery, algorithm = 'ball_tree')
    nn.fit(knnSpace)
    _, indices = nn.kneighbors(knnSpace)

    eIndex = []
    eLabel = []
    eFeatures = []

    for i in range(n):
        c = 0
        for j in indices[i]:
            if i == j:
                continue

            #if abs(layer[j] - layer[i]) > maxLayerJump:
                #continue
            #if (r[j] <= r[i]) and exOutward:
                #continue

            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]

            dr = r[j] - r[i]
            dphi = phi[j] - phi[i]
            dphi = np.arctan2(np.sin(dphi), np.cos(dphi))

            abs_dphi = abs(dphi)
            abs_dz = abs(dz)
            abs_dr = abs(dr)

            eucD = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            layerJump = abs(layer[j] - layer[i])

            if layerJump > maxLayerJump:
                continue
            if (r[j] <= r[i]) and exOutward:
                continue
            if (maxAbsDphi is not None) and (abs(dphi) > maxAbsDphi):
                continue

            dzOverDr = 0.0
            if maxAbsDzOverDr is not None:
                if abs(dr) < 1e-6:
                    continue
                dzOverDr = dz / dr
                if abs(dzOverDr) > maxAbsDzOverDr:
                    continue
            elif abs(dr) >= 1e-6:
                dzOverDr = dz / dr

            eIndex.append([i, j])
            eFeatures.append([dx, dy, dz, dr, dphi, eucD, abs_dphi, abs_dr, abs_dz, dzOverDr, float(layerJump)])
            eLabel.append(int((pid[i] == pid[j])and (pid[i] != 0)))

            c+=1
            if c >= k:
                break
    if len(eIndex) == 0:
        return (np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0, 11), dtype=np.float32))

    eIndex = np.array(eIndex, dtype=np.int64).T
    eLabel = np.array(eLabel, dtype=np.int64)
    eFeatures = np.array(eFeatures, dtype=np.float32)

    return eIndex, eLabel, eFeatures

