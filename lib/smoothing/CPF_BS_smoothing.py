#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:46:04 2018

@author: trang
"""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from lib.smoothing.additives import RMSE, sampling_discrete, resampling_sys
from tqdm import tqdm
import matplotlib.pyplot as plt #plot

def _CPF(y, X_cond, m, H, Q, R, xb, B, Nf,dx, time): 
        
    T  = len(time)
    Xa = np.zeros([dx, Nf, T+1])
    Xf = np.zeros([dx, Nf, T+1])
    mean_Xf = np.zeros([dx, Nf, T+1])
    cov_Xf = np.zeros([dx,dx, Nf, T+1])
    Wa = np.zeros([Nf, T+1])
    Wa[:,0] = 1/Nf*np.ones(Nf) 
    loglik = 0;
    # Initialize ensemble
    for i in range(Nf):
        Xa[:,i,0] =  np.random.multivariate_normal(xb,B)
    mean_Xf[...,0] = np.tile(xb,(Nf,1)).T
    Xf[...,0] = Xa[...,0]

    if np.all(X_cond) != set():
            Xa[:,-1,0] = X_cond[:,0] 
    for t in range(T):
        # Resampling
        ind = resampling_sys(Wa[:,t])
        ind = np.random.permutation(ind)
        # Forecasting
        xf, mean_xf, Q_xf, _ = m(Xa[...,t],time[t],ind)
        mean_Xf[...,t+1] = mean_xf; cov_Xf[...,t+1] = Q_xf
        Xa[...,t+1] = xf; Xf[...,t+1] = xf	
        # Replacing
        if np.all(X_cond) != set():
            Xa[:,-1,t+1] = X_cond[:,t+1] 

        # Weighting
        var_obs = np.where(~np.isnan(y[:,t]))[0]; 
        if len(var_obs) == 0:
            Wa[:,t+1] = 1/Nf*np.ones([Nf]) 
        else:
            Yx = H[var_obs,:].dot(Xa[...,t+1]) 
            innov = np.tile(y[var_obs,t], (Nf,1)).T - Yx; innov_end = (y[var_obs,t]- H[var_obs,:].dot(xf[:,-1]))[np.newaxis];
            const = np.sqrt((2*np.pi)**len(var_obs)*np.linalg.det(R[np.ix_(var_obs,var_obs)]))
            logwei = -.5 *np.sum(innov.T.dot(inv(R[np.ix_(var_obs,var_obs)]))*(innov.T),1) # for numerical stability
            logwei_end = -.5 *np.sum(innov_end.dot(inv(R[np.ix_(var_obs,var_obs)]))*(innov_end),1) # for numerical stability
            loglik += np.log((np.sum(np.exp(logwei[:-1])) +np.exp(logwei_end))/(const*Nf));
            Wa[:,t+1] = np.exp(logwei - np.max(logwei))/const#*Wa_prev
            Wa[:,t+1] = Wa[:,t+1]/np.sum(Wa[:,t+1])
            
  
    return Xa, Xf, mean_Xf, cov_Xf, Wa, ind, loglik

def _CPF_BS(Y,X,m,Q,H,R,xb,B,X_cond,dx, Nf, Ns,time):
    Xa, Xf, mean_Xf, cov_Xf, Wa, ind , loglik = _CPF(Y, X_cond, m, H, Q, R, xb, B, Nf,dx, time)
        
    T  = len(time)
    Xs = np.zeros([dx, Ns, T+1])
    Xf_mean= np.zeros([dx, Ns, T+1])
    Xf_cov= np.zeros([dx, dx, Ns, T+1])
    idx_smos = np.zeros([Ns, T+1])

    ind_smo = sampling_discrete(Wa[:,-1],Ns)
    Xs[...,-1] = Xa[:,ind_smo,-1]
    idx_smos[...,-1] = ind_smo

    for t in range(T-1,-1,-1):
        for i in range(Ns):
            innov = np.tile(Xs[:,i,t+1], (Nf,1)).T - mean_Xf[...,t+1]
            const = np.sqrt((2*np.pi)**dx*np.linalg.det(cov_Xf[...,t+1].transpose(-1,0,1)))
            logwei = -.5*np.sum(innov.T.dot(np.linalg.inv(cov_Xf[...,t+1].transpose(-1,0,1)))[np.arange(Nf),np.arange(Nf),:]*innov.T,1)
            wei_res = np.exp(logwei -np.max(logwei))*Wa[:,t]/const
            Ws = wei_res/np.sum(wei_res)
            ind_smo[i] = sampling_discrete(Ws,1)
        Xs[...,t] = Xa[:,ind_smo,t]
        Xf_mean[...,t+1] = mean_Xf[:,ind_smo,t+1]
        Xf_cov[...,t+1] = cov_Xf[...,ind_smo,t+1]
        idx_smos[...,t+1] = ind_smo
    return Xs, Xa, Xf, Xf_mean, Xf_cov, loglik, idx_smos



def CPF_BS(Y,X,m,Q,H,R,xb,B,Xcond,dx, Nf, Ns,time,nIter): 
    T  = len(time)
    Xcond_all = np.zeros([dx,T+1,nIter+1]);   Xf_all = np.zeros([dx,Nf,T+1,nIter+1]);
    Xa_all = np.zeros([dx,Nf,T+1,nIter+1]); Xs_all = np.zeros([dx,Ns,T+1,nIter+1]);
    Xcond_all[:,:,0] = Xcond
    Xs, Xa, Xf, Xf_mean,_, loglik = _CPF_BS(Y,X,m,Q,H,R,xb,B,Xcond,dx, Nf, Ns,time)
    Xf_all[:,:,:,0] = Xf; Xa_all[:,:,:,0] = Xa; Xs_all[:,:,:,0] = Xs
    class MSE:
        pre = 0; fil = 0; smo = 0
    
    Wd = 1/Ns*np.ones([Ns])    
    for i in tqdm(range(1,nIter+1)):
        ind_cond = np.random.choice(np.arange(0,Ns), 1, p=Wd)
        Xcond = np.squeeze(Xs[:,ind_cond,1:])
        Xs, Xa, Xf, Xfmean, _, loglik = _CPF_BS(Y,X,m,Q,H,R,xb,B,Xcond,dx, Nf, Ns,time)
        Xf_all[:,:,:,i] = Xf; Xa_all[:,:,:,i] = Xa; Xs_all[:,:,:,i] = Xs
        Xcond_all[:,:,i] = Xcond

        if i>5:
            MSE.fil += (RMSE(X[:,1:] - Xa[...,1:].mean(1)))**2
            MSE.smo += (RMSE(X[:,1:] - Xs[...,1:].mean(1)))**2
            MSE.pre += (RMSE(X[:,1:] - Xf[...,1:].mean(1)))**2

    out_CPF_BS= {'smoothed_samples': Xs_all,
               'conditioning_trajectory': Xcond_all,
               'filtered_samples': Xa_all,
               'forecasted_samples': Xf_all,
               'loglikelihood'    : loglik, #log_likelihood(Xf, Y, H, R),
               'RMSE'             : MSE,
               }
 
    return out_CPF_BS
