#!/usr/bin/env python
import numpy as np
from utils import MnistData


def optimal_system_size(input):
    singulars = np.linalg.svd(input, compute_uv=False)
    for i in range(1, len(singulars)):
        if singulars[i] < np.mean(singulars[:i]) / 2:
            print 'Optimal n for linear system would be', i
            return
    print 'Optimal n for linear system would be', len(singulars)


data = MnistData(50000, normalize=False)
(u, y) = data.get('trn', i=0)

# Test optimal system size
print 'In the beginning...'
optimal_system_size(u)

print 'After dimension-wise mean subtraction...'
u -= np.mean(u, axis=1, keepdims=True)
optimal_system_size(u)

print 'After variance normalization...'
u /= np.maximum(np.std(u, axis=1, keepdims=True), 1e-10)
optimal_system_size(u)
