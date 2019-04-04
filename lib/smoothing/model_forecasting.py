
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:19:34 2018

@author: trang
"""


import numpy as np

def m_true(x,pos,ind, Q, f,jacF, dt_model):
    dx,N = np.shape(x)
    xf = np.zeros([dx,N]);
    mean_xf = np.zeros([dx,N]);
    M_xf = np.zeros([dx,dx,N]);
    Q_xf = np.zeros([dx,dx,N]);
    #f = lambda x: l63_predict(x, dt, sigma, rho, beta) # python version (slow)
    XX = x
    for j in range(dt_model):
        XX = f(XX)
    noise = np.zeros([dx,N])
    for i in range(N):
        noise[:,i] = np.random.multivariate_normal(np.zeros(dx),Q)
    if len(ind)==1:
        xf  = XX + noise
    else:
        xf  = XX[:,ind] + noise
    mean_xf = XX
    for i in range(N):
        M_xf[:,:,i] = jacF(x[:,i])
        Q_xf[:,:,i] = Q
    return xf, mean_xf, Q_xf, M_xf
    
