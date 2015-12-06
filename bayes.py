#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

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

wavelen = np.loadtxt("espectro.dat", usecols=(0,))
frec = np.loadtxt("espectro.dat", usecols=(1,))
data = np.zeros((2, len(wavelen)))
data[0] = wavelen
data[1] = (10**16)*frec

beta0_grid, beta1_grid = np.mgrid[0:2:99j, 0:10:99j]
beta_prior_pars = [1, 0.4, 5, 2]
prior_grid = fill_prior1(beta0_grid, beta1_grid, beta_prior_pars)
