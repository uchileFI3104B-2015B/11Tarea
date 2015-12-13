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
Este script modela 2 parametros a la vez a partir de los datos de
espectro.dat, al plotear se observa un segmento del espectro de una fuente
que muesta continuo y una linea de absorcion, el continuo esta dado y la linea
de absorcion se modela con una gaussiana
'''
plt.style.use('bmh')
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['figure.figsize'] = '7, 5'

def make_figure_axes(x, y, fig_number=1, fig_size=8):
    '''
    Crea un set de 3 ejes para plotear un grafico 2D funcion + marginals
    '''
    # determine max size
    size_x = x.max() - x.min()
    size_y = y.max() - y.min()
    max_size = max(size_x, size_y)
    min_size = min(size_x, size_y)

    if size_x >= size_y:
        fig_size_x = fig_size
        fig_size_y = (0.12 * fig_size_x +
                      0.65 * fig_size_x * min_size / max_size +
                      0.02 * fig_size_x +
                      0.18 * fig_size_x +
                      0.03 * fig_size_x)
        rect_main = [0.12,
                     0.12 * fig_size_x / fig_size_y,
                     0.65,
                     0.65 * fig_size_x * min_size / max_size / fig_size_y]
        rect_x = [0.12, ((0.12 + 0.65 * min_size / max_size + 0.02) *
                         fig_size_x / fig_size_y),
                  0.65, 0.18 * fig_size_x / fig_size_y]
        rect_y = [0.79, 0.12 * fig_size_x / fig_size_y,
                  0.18, 0.65 * fig_size_x * min_size / max_size / fig_size_y]
    else:
        fig_size_y = fig_size
        fig_size_x = (0.12 * fig_size_y +
                      0.65 * fig_size_y * min_size / max_size +
                      0.02 * fig_size_y +
                      0.18 * fig_size_y +
                      0.03 * fig_size_y)
        rect_main = [0.12 * fig_size_y / fig_size_x,
                     0.12,
                     0.65 * fig_size_y * min_size / max_size / fig_size_x,
                     0.65]
        rect_x = [0.12 * fig_size_y / fig_size_x, 0.79,
                  0.65 * fig_size_y * min_size / max_size / fig_size_x, 0.18]
        rect_y = [((0.12 + 0.65 * min_size / max_size + 0.02) *
                   fig_size_y / fig_size_x), 0.12,
                  0.18 * fig_size_y / fig_size_x, 0.65]

    fig = plt.figure(fig_number, figsize=(fig_size_x, fig_size_y))
    fig.clf()

    ax_main = fig.add_axes(rect_main)
    ax_marginal_x = fig.add_axes(rect_x, xticklabels=[])
    ax_marginal_y = fig.add_axes(rect_y, yticklabels=[])

    return ax_main, ax_marginal_x, ax_marginal_y


def plot_distribution(x, y, z, cmap='PuBu_r'):
    x_limits = (x.min(), x.max())
    y_limits = (y.min(), y.max())

    ax_main, ax_marginal_x, ax_marginal_y = make_figure_axes(x, y)
    ax_main.pcolormesh(x, y, z, cmap=cmap)

    marginal_x = np.sum(z, axis=1)
    ax_marginal_x.plot(x[:,0], marginal_x)
    [l.set_rotation(-90) for l in ax_marginal_y.get_xticklabels()]

    marginal_y = np.sum(z, axis=0)
    ax_marginal_y.plot(marginal_y, y[0])

    ax_main.set_xlim(x_limits)
    ax_main.set_ylim(y_limits)

    ax_marginal_x.set_xlim(x_limits)
    ax_marginal_y.set_ylim(y_limits)
    return ax_main, ax_marginal_x, ax_marginal_y

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
    return 1e-16 - A * scipy.stats.norm(loc=6563, scale=sigma).pdf(x)


def prior(beta, params):
    '''
    Retorna la densidad de probabilidad para cierto modelo evaluada en
    un punto
    '''
    if len(beta)==2:
        beta0, beta1 = beta
        mu0, sigma0, mu1, sigma1 = params
        S = -1. / 2 * ((beta0 - mu0)**2 / sigma0**2 + (beta1 - mu1)**2 / sigma1**2)
        P = np.exp(S) / (2 * np.pi * sigma0 * sigma1)
    elif(beta)==4
        beta0, beta1, beta2, beta3 = beta
        mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
        S0 = -1. / 2 * (beta0 - mu0)**2 / sigma0**2
        S1 = -1. / 2 * (beta1 - mu1)**2 / sigma1**2
        S2 = -1. / 2 * (beta2 - mu2)**2 / sigma2**2
        S3 = -1. / 2 * (beta3 - mu3)**2 / sigma3**2
        P = np.exp(S0 + S1 + S2 + S3)
        P = P / (2 * np.pi * sigma0 * sigma1 * sigma2 * sigma3)
    return P

def fill_prior(beta0_grid, beta1_grid, prior_params):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = prior([beta0_grid[i,j], beta1_grid[i,j]], prior_params)
    return output

def likelihood(beta, data, modelo):
    '''
    Calcula verosimilitud
    '''
    beta0, beta1 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - modelo(x, *beta))**2)
    L = (2 * np.pi * 1.5**2)**(-N / 2.) * np.exp(-S / 2 / 1.5**2)
    return L

def fill_likelihood(beta0_grid, beta1_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = likelihood([beta0_grid[i,j], beta1_grid[i,j]], data)
    return output

# Main

# Inicializa
x_sample, y_sample = datos()
x = np.linspace(min(x_sample), max(x_sample))

# Grilla suficientemente grande
beta0_grid, beta1_grid = np.mgrid[-0.0001:0.0001:201j, -5:15:201j]
dx = 0.0002 / 200
dy = 20 / 100

prior_params = [0.08e-16, 100, 3, 100]
prior_grid = fill_prior(beta0_grid, beta1_grid, prior_params)
likelihood_grid = fill_likelihood(beta0_grid, beta1_grid, [x_sample, y_sample])

ax_main, ax_0, ax_1 = plot_distribution(beta0_grid, beta1_grid, likelihood_grid * prior_grid)
