#!/usr/bin/env python

""" AnDA_stat_functions.py: Collection of statistical functions used in AnDA. """

__author__ = "Pierre Tandeo and Phi Huynh Viet"
__version__ = "1.0"
__date__ = "2016-10-16"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@telecom-bretagne.eu"

import numpy as np



def RMSE(error):
  return np.sqrt(np.mean(error**2))

def sampling_discrete(W, m):  ### Discrete sampling, ADDED by TRANG ###
    "Returns m indices given N weights"
    cumprob = np.cumsum(W)
    n = np.size(cumprob)
    R = np.random.rand(m)
    ind = np.zeros(m)
    for i in range(n):
        ind += R> cumprob[i]
    ind = np.array(ind, dtype = int)    
    return ind


def resampling_sys(W): ### systematic resampling with respect to multinomial distribution, ADDED by TRANG ###
    "Returns a N-set of indices given N weights"
    N = np.size(W)
    u0 = np.random.rand(1)
    u = (range(N) +u0).T/N;
    qc = np.cumsum(W)
    qc = qc[:]
    qc = qc/qc[-1]
    new= np.concatenate((u,qc), axis=0)
    ind1 = np.argsort(new)
#   ind2 = np.where(ind1<=N-1);
    ind2 = np.array(np.where(ind1<=N-1),dtype = int)
    ind = ind2- range(N)
    a = ind[0,]
    return a

def inv_svd(A):
    "Returns the inverse matrix by SVD"
    U, s, V = np.linalg.svd(A, full_matrices=True)
    invs = 1./s
    n = np.size(s)
    invS = np.zeros((n,n))
    invS[:n, :n] = np.diag(invs)
    invA=np.dot(V.T,np.dot(invS, U.T))
    return invA
