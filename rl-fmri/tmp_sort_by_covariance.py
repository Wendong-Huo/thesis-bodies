import numpy as np
N = 5
t_org = np.arange(N)
pi = np.random.permutation(N)
x_org = np.array([np.random.randn(100)*k for k in range(N)])
S_org = np.cov(x_org)
print("Covariance of sorted time steps", S_org, sep="\n")

t_obs = t_org[pi]
x_obs = x_org[pi]
S_obs = np.cov(x_obs)
print("Covariance of unsorted time steps", S_obs, sep="\n")

#%% Using indexing S[p][:,p]
p = np.argsort(t_obs)
print("Reconstruction equals original:", S_obs[p][:,p] == S_org, sep="\n")

#%% Alternative using Permutation matrix
P = np.eye(N)[p]
print("Permutation matrix", P, sep="\n")
print("Reconstruction equals original:", P@S_obs@P.T == S_org, sep="\n")