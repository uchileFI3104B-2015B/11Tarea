# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy.stats import kstest
from IPython.display import Image


'''
Este script modela 2 y 4 parametros a la vez a partir de los datos de
espectro.dat, al plotear se observa un segmento del espectro de una fuente
que muesta continuo y una linea de absorcion, el continuo esta dado y la linea
de absorcion se modela con una o doble gaussiana
'''


def datos():
    '''
    Descarga los datos de un archivo y los retorna en columnas
    '''
    lambda1 = np.loadtxt('espectro.dat', usecols=(-2,))
    flujo = np.loadtxt('espectro.dat', usecols=(-1,))
    return lambda1, flujo

def gaussiana(x, A, sigma):
    '''
    modela linea de absorcion a traves de una gaussiana
    '''
    return 1e-16 - (A * scipy.stats.norm(loc=6563, scale=sigma).pdf(x)) * 0.08


def gaussiana2(x, A, A2, sigma, sigma2):
    '''
    modela linea de absorsion a traves de doble gaussiana
    '''
    return 1e-16 - (A * scipy.stats.cauchy(loc=6563, scale=sigma).pdf(x) +
            A2 * scipy.stats.cauchy(loc=6563, scale=sigma2).pdf(x))


def prior(beta, params):
    '''
    Retorna la densidad de probabilidad para cierto modelo evaluada en
    un punto
    '''
    if len(beta)==2:
        beta0, beta1 = beta
        mu0, sigma0, mu1, sigma1 = params
        S = -1. / 2 * ((beta0 - mu0)**2 / sigma0**2 +
                       (beta1 - mu1)**2 / sigma1**2)
        P = np.exp(S) / (2 * np.pi * sigma0 * sigma1)
    elif len(beta)==4:
        beta0, beta1, beta2, beta3 = beta
        mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
        S0 = -1. / 2 * (beta0 - mu0)**2 / sigma0**2
        S1 = -1. / 2 * (beta1 - mu1)**2 / sigma1**2
        S2 = -1. / 2 * (beta2 - mu2)**2 / sigma2**2
        S3 = -1. / 2 * (beta3 - mu3)**2 / sigma3**2
        SS = np.exp(S0 + S1 + S2 + S3)
        P = SS / (2 * np.pi * sigma0 * sigma1 * sigma2 * sigma3)
    return P

def fill_prior_1(beta0_grid, beta1_grid, prior_params):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = prior([beta0_grid[i,j], beta1_grid[i,j]], prior_params)
    return output


def fill_prior_2(beta0_grid, beta1_grid, beta2_grid, beta3_grid, prior_params,
                 data):
    output = np.zeros(beta0_grid.shape)
    ni, nj, nk, nl = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for l in range(nl):
                    output[i, j, k, l] = prior([beta0_grid[i, j, k, l],
                                               beta1_grid[i, j, k, l],
                                               beta2_grid[i, j, k, l],
                                               beta3_grid[i, j, k, l]],
                                               prior_params)
    return output

def likelihood(beta, data, modelo):
    '''
    Calcula verosimilitud
    '''
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - modelo(x, *beta))**2)
    L = (2 * np.pi * 1.5**2)**(-N / 2.) * np.exp(-S / 2 / 1.5**2)
    return L

def fill_likelihood_m1(beta0_grid, beta1_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = likelihood([beta0_grid[i,j], beta1_grid[i,j]],
                                      data, gaussiana)
    return output



# Main

# Inicializa
x_sample, y_sample = datos()
x = np.linspace(min(x_sample), max(x_sample))

# Grilla suficientemente grande
beta0_grid, beta1_grid = np.mgrid[-0.0001:0.0001:201j, -5:15:201j]
dx1 = 0.0002 / 200
dy1 = 20 / 200

# Modelo 1
prior_params = [0.02e-16, 10, 3, 10]
prior_grid_m1 = fill_prior_1(beta0_grid, beta1_grid, prior_params)
likelihood_grid_m1 = fill_likelihood_m1(beta0_grid, beta1_grid, [x_sample, y_sample])

post_grid1 = likelihood_grid_m1 * prior_grid_m1
P_E1 = np.sum(post_grid1) * dx1 * dy1
marg_A_mod1 = np.sum(post_grid1, axis=1) * dy1 / P_E1
marg_sigma_mod1 = np.sum(post_grid1, axis=0) * dx1 / P_E1

# Calculo parametros
E_A_mod1 = np.sum(beta0_grid[:, 0] * marg_A_mod1) * dx1
E_sigma_mod1 = np.sum(beta1_grid[0, :] * marg_sigma_mod1) * dy1

print 'Primer modelo: Gaussiana simple'
print 'Amplitud                :', E_A_mod1
print 'sigma                   :', E_sigma_mod1
print ''

# Modelo 2
prior_params2 = [0.02e-16, 0.04e-16, 6, 3]
