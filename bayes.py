#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

def prior1(params, prior_pars):
    param0, param1 = params
    mu0, sigma0, mu1, sigma1 = prior_pars
    S = -1/2.0 * ((param0 - mu0)**2 / sigma0**2 + (param1 - mu1)**2 / sigma1**2)
    P = np.exp(S) / (2 * np.pi * sigma0 * sigma1)
    return P

def fill_prior1(param0_grid, param1_grid, prior_pars):
    out = np.zeros(param0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            out[i, j] = prior1([param0_grid[i,j], param1_grid[i,j]], prior_pars)
    return out

def likelihood(beta, data):
    beta0, beta1 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - 1 + beta0 * gaussian(x, 6563, beta1) )**2)
    L = (2 * np.pi * noise**2)**(-N / 2.) * np.exp(-S / (2*noise**2))
    return L

def fill_likelihood(beta0_grid, beta1_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = likelihood([beta0_grid[i,j], beta1_grid[i,j]], data)
    return output


wavelen = np.loadtxt("espectro.dat", usecols=(0,))
frec = np.loadtxt("espectro.dat", usecols=(1,))
data = np.zeros((2, len(wavelen)))
data[0] = wavelen
data[1] = (10**16)*frec

s = 0
n = 0
for i in range(len(wavelen)):
    if data[0][i] <= 6530 or data[0][i]>= 6600:
        s += (data[1][i] - 1)**2
        n += 1
noise = np.sqrt(s/(1.0*n))

beta0_grid, beta1_grid = np.mgrid[0.6:0.9:99j, 3:4.5:99j]
beta_prior_pars = [1, 0.4, 5, 2]
prior_grid = fill_prior1(beta0_grid, beta1_grid, beta_prior_pars)
likelihood_grid = fill_likelihood(beta0_grid, beta1_grid, [data[0], data[1]])
post_grid = likelihood_grid * prior_grid
plt.clf()
plt.pcolormesh(beta0_grid, beta1_grid, post_grid)
plt.show()

# pars optimos segun esto: 0.764, 3.712
