"""
Available on https://github.com/ptandeo/CEDA
@author: Pierre Tandeo

"""

import numpy as np
from numpy.linalg import inv
from lib.smoothing.additives import RMSE, inv_svd



def _EnKF(dx, T, dy, xb, B, Q, R, Ne, alpha, f, H, obs):
  Xa = np.zeros([dx, Ne, T+1])
  Xf = np.zeros([dx, Ne, T])

  # Initialize ensemble
  for i in range(Ne):
    Xa[:,i,0] = np.random.multivariate_normal(xb,B)

  for t in range(T):
    # Forecast
    # for i in range(Ne):
    #   Xf[:,i,t] = f(Xa[:,i,t]) + sqQ.dot(prng.normal(size=dx))
    xf,_, _, _ = f(Xa[...,t],t,np.ones((1)))
    Xf[:,:,t] = xf
    # Update
    # Update
    var_obs = np.where(~np.isnan(obs[:,t]))[0];
    dy = len(var_obs)
    if dy == 0:
      Xa[:,:,t+1] = Xf[:,:,t]
    else:
      Y = H[var_obs,:].dot(Xf[:,:,t]) + (np.random.multivariate_normal(np.zeros((dy)),R[np.ix_(var_obs,var_obs)],Ne)).T
      Pfxx = np.cov(Xf[:,:,t])
      K = Pfxx.dot(H[var_obs,:].T).dot(inv_svd(H[var_obs,:].dot(Pfxx).dot(H[var_obs,:].T) + R[np.ix_(var_obs,var_obs)]/alpha))
      innov = np.tile(obs[var_obs,t], (Ne, 1)).T - Y
      Xa[:,:,t+1] = Xf[:,:,t] + K.dot(innov)
#      for i in range(Ne):
#        innov = obs[:,t] - Y[:,i]
#        Xa[:,i,t+1] = Xf[:,i,t] + K.dot(innov)

  return Xa, Xf

def _EnKS(dx, Ne, T, H, R, Y, Xt, dy, xb, B, Q, alpha, f):
  Xa, Xf = _EnKF(dx, T, dy, xb, B, Q, R, Ne, alpha, f, H, Y)

  Xs = np.zeros([dx, Ne, T+1])
  Xs[:,:,-1] = Xa[:,:,-1]
  for t in range(T-1,-1,-1):
    Paf = np.cov(Xa[:,:,t], Xf[:,:,t])[:dx, dx:] ### MODIF PIERRE ###
    Pff = np.cov(Xf[:,:,t])
    try:
      K = Paf.dot(inv(Pff))
    except:
      K = Paf.dot(Pff**(-1)) ### MODIF PIERRE ###
    Xs[:,:,t] = Xa[:,:,t] + K.dot(Xs[:,:,t+1] - Xf[:,:,t])
   # for i in range(Ne):
   #   Xs[:,i, t] = Xa[:5,i,t] + K.dot(Xs[:,i,t+1] - Xf[:,i,t])

  return Xs, Xa, Xf

def EnKS(Y,Xt,Q,f,H,R,xb,B,dx,dy,Ne,T,alpha):
  Xs, Xa, Xf = _EnKS(dx, Ne, T, H, R, Y, Xt, dy, xb, B, Q, alpha, f)

  res = {
          'smoothed_ensemble': Xs,
          'analysis_ensemble': Xa,
          'forecast_ensemble': Xf,
          'RMSE'             : RMSE(Xt[:,1:] - Xs[:,:,1:].mean(1))
         }
  return res


