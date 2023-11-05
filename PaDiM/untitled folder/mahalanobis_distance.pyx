# mahalanobis_distance.pyx
import numpy as np
cimport numpy as cnp

def mahalanobis_distance(cnp.ndarray[float, ndim=1] sample,
                         cnp.ndarray[float, ndim=1] mean,
                         cnp.ndarray[float, ndim=2] cov_inv):
    cdef cnp.ndarray[float, ndim=1] diff = sample - mean
    cdef float dist = np.sqrt(np.dot(diff.T, np.dot(cov_inv, diff)))
    return dist
