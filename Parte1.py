# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:54:07 2015

@author: splatt
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['figure.figsize'] = '7, 5'

def make_figure_axes(x, y, fig_number=1, fig_size=8):
    '''
    Creates a set of 3 axes to plot 2D function + marginals
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


def plot_distribution(x, y, z, cmap='viridis'):
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


def Gauss2D(x, y, mu, sigma_pars):
    mu_x, mu_y = mu
    sigma_x, sigma_y, rho = sigma_pars
    A = 1 / (2 * np.pi * 1. * 1. * np.sqrt(1-0.5**2))
    B = np.exp(-((x - mu_x)**2 / sigma_x**2 + (y - mu_y)**2 / sigma_y**2 - 
                 2 * rho * (x - mu_x) * (y - mu_y) / sigma_x / sigma_y) / 
                 (2 * (1 - rho**2)) )
    return A * B


def prior(beta, params):
    beta0, beta1 = beta
    mu0, sigma0, mu1, sigma1 = params
    S = -1. / 2 * ((beta0 - mu0)**2 / sigma0**2 + (beta1 - mu1)**2 / sigma1**2)
    P = np.exp(S) / (2 * np.pi * sigma0 * sigma1)
    return P

def fill_prior(beta0_grid, beta1_grid, prior_params):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = prior([beta0_grid[i,j], 
                                 beta1_grid[i,j]], prior_params)
    return output

def likelihood(beta, data):
    beta0, beta1 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - beta0 - beta1 * x)**2)
    L = (2 * np.pi * 1.5**2)**(-N / 2.) * np.exp(-S / 2 / 1.5**2)
    return L

def fill_likelihood(beta0_grid, beta1_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = likelihood([beta0_grid[i,j], beta1_grid[i,j]], data)
    return output

datos = np.loadtxt('espectro.dat')

w = datos[:, 0]
fnu = datos[:, 1]
fnu=fnu*(10**18)
plt.figure(0)
plt.plot(w,fnu)
plt.xlabel('Longitud de onda w [Angstrom]')
plt.ylabel('Flujo por unidad de freuencia [erg / s / Hz / cm^2 * 10^18]')

x, y = np.mgrid[6000:7000:50j, 50:200:50j]
    
ni, nj = x.shape
g2d = np.zeros((ni, nj))
mu = [np.mean(w), np.mean(fnu)]
sigma_pars = [np.std(w), np.std(fnu), 0]
for i in range(ni):
    for j in range(nj):
        g2d[i,j] = Gauss2D(x[i,0], y[0,j], mu, sigma_pars)

plt.figure(1)
ax1, ax2, ax3 = plot_distribution(x, y, g2d)
ax1.axvline(mu[0], lw=6563, color='w')
ax1.axhline(mu[1], lw=6563, color='w')
ax2.axvline(mu[0], lw= 100, color='k')
ax3.axhline(mu[1], lw= 100, color='k')

beta0_grid, beta1_grid = np.mgrid[6000:7000:50j, 50:200:50j]
beta_prior_pars = [6563, 6460, 100, 91]
prior_grid = fill_prior(beta0_grid, beta1_grid, beta_prior_pars)

plt.figure(2)
plt.pcolormesh(beta0_grid, beta1_grid, prior_grid, cmap='viridis')
plt.gca().set_aspect('equal')

likelihood_grid = fill_likelihood(beta0_grid, beta1_grid, [w, fnu])
post_grid = likelihood_grid * prior_grid

plt.figure(3)
ax1, ax2, ax3 = plot_distribution(beta0_grid, beta1_grid, post_grid)
ax1.axvline(2.5, lw=0.5, color='w')
ax1.axhline(3.4, lw=0.5, color='w')
ax2.axvline(2.5, lw=1., color='k')
ax3.axhline(3.4, lw=1., color='k')

beta0_grid, beta1_grid = np.mgrid[-5:11:50j, 1:6:50j]
dx = 16 / 200
dy = 5 / 100

prior_grid = fill_prior(beta0_grid, beta1_grid, [1, 100, 10, 100])
likelihood_grid = fill_likelihood(beta0_grid, beta1_grid, [w, fnu])

plt.figure(4)
plt.plot_distribution(beta0_grid, beta1_grid, likelihood_grid * prior_grid)
