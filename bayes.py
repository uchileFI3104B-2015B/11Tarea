#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

def prior1(params, prior_pars):
    Amp, std = params
    mu0, sigma0, mu1, sigma1 = prior_pars
    S = -1/2.0 * ((Amp - mu0)**2 / sigma0**2 + (std - mu1)**2 / sigma1**2)
    P = np.exp(S) / (2 * np.pi * sigma0 * sigma1)
    return P

def prior2(params, prior_pars):
    Amp1, std1, Amp2, std2 = params
    mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3 = prior_pars
    S = -1/2.0 * ((Amp1 - mu0)**2 / sigma0**2 + (std1 - mu1)**2 / sigma1**2 +
                  (Amp2 - mu2)**2 / sigma2**2 + (std2 - mu3)**2 / sigma3**2)
    P = np.exp(S) / ((2*np.pi)**2 * sigma0 * sigma1 * sigma2 * sigma3)
    return P

def fill_prior1(param0_grid, param1_grid, prior_pars):
    out = np.zeros(param0_grid.shape)
    ni, nj = param0_grid.shape
    for i in range(ni):
        for j in range(nj):
            out[i, j] = prior1([param0_grid[i,j], param1_grid[i,j]], prior_pars)
    return out

def fill_prior2(par0_grid, par1_grid, par2_grid, par3_grid, prior_pars):
    out = np.zeros(par0_grid.shape)
    ni, nj, nk, nl = par0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for l in range(nl):
                    out[i, j, k ,l] = prior2([par0_grid[i, j, k, l],
                                              par1_grid[i, j, k, l],
                                              par2_grid[i, j, k, l],
                                              par3_grid[i, j, k, l]],
                                             prior_pars)
    return out

def likelihood1(beta, data):
    beta0, beta1 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - lamb*100 + beta0 * gaussian(x, 6563, beta1) )**2)
    L = (2 * np.pi * noise**2)**(-N / 2.) * np.exp(-S / (2*noise**2))
    return L

def likelihood2(beta, data):
    beta0, beta1, beta2, beta3 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - lamb*100 + beta0 * gaussian(x, 6563, beta1) +
                beta2 * gaussian(x, 6563, beta3) )**2)
    L = (2 * np.pi * noise**2)**(-N / 2.) * np.exp(-S / (2*noise**2))
    return L

def fill_likelihood1(beta0_grid, beta1_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = likelihood1([beta0_grid[i,j], beta1_grid[i,j]], data)
    return output

def fill_likelihood2(beta0_grid, beta1_grid, beta2_grid, beta3_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj, nk, nl = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for l in range(nl):
                    output[i, j, k, l] = likelihood2([beta0_grid[i, j, k, l],
                                                      beta1_grid[i, j, k, l],
                                                      beta2_grid[i, j, k, l],
                                                      beta3_grid[i, j, k, l]],
                                                     data)
    return output

# Lectura de datos
lamb = 0.4
wavelen = np.loadtxt("espectro.dat", usecols=(0,))
frec = np.loadtxt("espectro.dat", usecols=(1,))
data = np.zeros((2, len(wavelen)))
data[0] = wavelen
data[1] = (lamb * 10**18)*frec

# Determinacion de ruido
s = 0
n = 0
for i in range(len(wavelen)):
    if data[0][i] <= 6530 or data[0][i]>= 6600:
        s += (data[1][i] - 100 * lamb)**2
        n += 1
noise = np.sqrt(s/(1.0*n))

# Modelo 1
beta0_grid1, beta1_grid1 = np.mgrid[(70*lamb):(84*lamb):201j, 3.4:4.0:201j]
beta_prior_pars1 = [80*lamb, 40*lamb, 3.6, 3]
prior_grid1 = fill_prior1(beta0_grid1, beta1_grid1, beta_prior_pars1)
likelihood_grid1 = fill_likelihood1(beta0_grid1, beta1_grid1,
                                    [data[0], data[1]])
post_grid1 = likelihood_grid1 * prior_grid1
dx1 = 14.0*lamb / 200
dy1 = 0.6 / 200
P_E1 = np.sum(post_grid1) * dx1 * dy1
marg_Amp1 = np.sum(post_grid1, axis=1) * dy1 / P_E1
marg_std1 = np.sum(post_grid1, axis=0) * dx1 / P_E1
EAmp1 = np.sum(beta0_grid1[:, 0] * marg_Amp1) * dx1
Estd1 = np.sum(beta1_grid1[0, :] * marg_std1) * dy1

# Modelo 2
l = lamb
beta0_grid2, beta1_grid2, beta2_grid2, beta3_grid2 = np.mgrid[8:24:101j,
                                                              1.0:4.0:101j,
                                                              14:26:101j,
                                                              5.5:11.5:101j]
beta_prior_pars2 = [40*lamb, 20*lamb, 2.5, 1, 50*lamb, 25*lamb, 8.5, 3]
prior_grid2 = fill_prior2(beta0_grid2, beta1_grid2, beta2_grid2,
                          beta3_grid2, beta_prior_pars2)
likelihood_grid2 = fill_likelihood2(beta0_grid2, beta1_grid2, beta2_grid2,
                                    beta3_grid2, [data[0], data[1]])
post_grid2 = likelihood_grid2 * prior_grid2
dx2 = 16.0 / 100
dy2 = 3.0 / 100
dz2 = 8.0 / 100
dw2 = 6.0 / 100
P_E2 = np.sum(post_grid2) * dx2 * dy2 * dz2 * dw2
marg_Amp21 = (np.sum(np.sum(np.sum(post_grid2, axis=1), axis=1), axis=1) *
              dy2 * dz2 * dw2 / P_E2)
marg_std21 = (np.sum(np.sum(np.sum(post_grid2, axis=0), axis=1), axis=1) *
              dx2 * dz2 * dw2 / P_E2)
marg_Amp22 = (np.sum(np.sum(np.sum(post_grid2, axis=0), axis=0), axis=1) *
              dx2 * dy2 * dw2 / P_E2)
marg_std22 = (np.sum(np.sum(np.sum(post_grid2, axis=0), axis=0), axis=0) *
              dx2 * dy2 * dz2 / P_E2)
EAmp21 = np.sum(beta0_grid2[:, 0, 0, 0] * marg_Amp21) * dx2
Estd21 = np.sum(beta1_grid2[0, :, 0, 0] * marg_std21) * dy2
EAmp22 = np.sum(beta2_grid2[0, 0, :, 0] * marg_Amp22) * dz2
Estd22 = np.sum(beta3_grid2[0, 0, 0, :] * marg_std22) * dw2

print "Factor bayesiano:", P_E1/P_E2

plt.clf()
# Plots modelo 1
plt.figure(1)
plt.pcolormesh(beta0_grid1, beta1_grid1, post_grid1)
plt.figure(2)
plt.plot(beta0_grid1[:, 0], marg_Amp1)
plt.figure(3)
plt.plot(beta1_grid1[0, :], marg_std1)
# Plots modelo 2
plt.figure(4)
plt.plot(beta0_grid2[:, 0, 0, 0], marg_Amp21)
plt.figure(5)
plt.plot(beta1_grid2[0, :, 0, 0], marg_std21)
plt.figure(6)
plt.plot(beta2_grid2[0, 0, :, 0], marg_Amp22)
plt.figure(7)
plt.plot(beta3_grid2[0, 0, 0, :], marg_std22)
plt.show()

# opt params: 0.4, 2.5, 0.5, 8.5
