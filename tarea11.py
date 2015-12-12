#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit

'''
En este código se busca modelar el ensanchamiento de una linea de absorcion
en un espectro de radiación, mediante técnicas bayesianas.
Para ello se utilizan dos modelos: uno gaussiano simple y otro con dos
funciones gaussianas.
'''


def modelo_1(x, A, sigma):
    '''
    Retorna la función correspondiente al primer modelo, es decir, una
    gaussiana trasladada en 1e-16
    '''
    return 1e-16 - A * scipy.stats.cauchy(loc=6563, scale=sigma).pdf(x)


def modelo_2(x, Amplitud1, sigma1, Amplitud2, sigma2):
    '''
    Retorna la función correspondiente al primer modelo, es decir, la suma
    de dos gaussianas trasladada en 1e-16
    '''
    gauss1 = (Amplitud1 * scipy.stats.norm(loc=6563, scale=sigma1).pdf(x))
    gauss2 = (Amplitud2 * scipy.stats.norm(loc=6563, scale=sigma2).pdf(x))
    return 1e-16 - gauss1 - gauss2


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


def plot_distribution(x, y, z, cmap='PuBu_r'):
    x_limits = (x.min(), x.max())
    y_limits = (y.min(), y.max())

    ax_main, ax_marginal_x, ax_marginal_y = make_figure_axes(x, y)
    ax_main.pcolormesh(x, y, z, cmap=cmap)

    marginal_x = np.sum(z, axis=1)
    ax_marginal_x.plot(x[:, 0], marginal_x)
    [l.set_rotation(-90) for l in ax_marginal_y.get_xticklabels()]

    marginal_y = np.sum(z, axis=0)
    ax_marginal_y.plot(marginal_y, y[0])

    ax_main.set_xlim(x_limits)
    ax_main.set_ylim(y_limits)

    ax_marginal_x.set_xlim(x_limits)
    ax_marginal_y.set_ylim(y_limits)
    return ax_main, ax_marginal_x, ax_marginal_y


def likelihood(data, params, modelo):
    '''
    Calcula la verosimilitud para cierto modelo
    '''
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - modelo(x, *params))**2)
    L = (2 * np.pi * 1.5**2)**(-N / 2.) * np.exp(-S / 2 / 1.5**2)
    return L


def prior(beta, params):
    '''
    Retorna la densidad de probabilidad para cierto modelo evaluada en
    un punto
    '''
    if len(beta) == 2:
        beta0, beta1 = beta
        mu0, sigma0, mu1, sigma1 = params
        S0 = -1. / 2 * (beta0 - mu0)**2 / sigma0**2
        S1 = -1. / 2 * (beta1 - mu1)**2 / sigma1**2
        P = np.exp(S0 + S1) / (2 * np.pi * sigma0 * sigma1)
    elif len(beta) == 4:
        beta0, beta1, beta2, beta3 = beta
        mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
        S0 = -1. / 2 * (beta0 - mu0)**2 / sigma0**2
        S1 = -1. / 2 * (beta1 - mu1)**2 / sigma1**2
        S2 = -1. / 2 * (beta2 - mu2)**2 / sigma2**2
        S3 = -1. / 2 * (beta3 - mu3)**2 / sigma3**2
        P = np.exp(S0 + S1 + S2 + S3)
        P = P / (2 * np.pi * sigma0 * sigma1 * sigma2 * sigma3)
    else:
        return False
    return P


def fill_prior_1(beta0_grid, beta1_grid, prior_params, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = prior([beta0_grid[i, j], beta1_grid[i, j]],
                                 prior_params)
            likelihood_m1[i, j] = likelihood([data[0], data[1]],
                                             [beta0_grid[i, j],
                                              beta1_grid[i, j]], modelo_1)
    return output, likelihood_m1


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
                    likelihood_m1[i, j] = likelihood([data[0], data[1]],
                                                     [beta0_grid[i, j, k, l],
                                                      beta1_grid[i, j, k, l],
                                                      beta2_grid[i, j, k, l],
                                                      beta3_grid[i, j, k, l]],
                                                     modelo_2)
    return output, likelihood_m1

# Main
# Datos
wavelength = np.loadtxt("espectro.dat", usecols=[0])
Fnu = np.loadtxt("espectro.dat", usecols=[1])
x = np.linspace(min(wavelength), max(wavelength), 100)

# Modelo 1:
Amplitud_mod1 = Fnu.max() - Fnu.min()
sigma_mod1 = 6  # Del gráfico
adivinanza_mod1 = [Amplitud_mod1, 0.1e-16, sigma_mod1, 2]

beta0_grid, beta1_grid = np.mgrid[-0.0001:0.0001:201j, -5:15:201j]
n0, n1 = beta0_grid.shape
prior_m1 = np.zeros((n0, n1))
likelihood_m1 = np.zeros((n0, n1))
prior_m1, likelihood_m1 = fill_prior_1(beta0_grid, beta1_grid, adivinanza_mod1,
                                       [wavelength, Fnu])

post_grid1 = likelihood_m1 * prior_m1
dx1 = 0.0002 / 200
dy1 = 20 / 200
P_E1 = np.sum(post_grid1) * dx1 * dy1
marg_Amplitud_mod1 = np.sum(post_grid1, axis=1) * dy1 / P_E1
marg_sigma_mod1 = np.sum(post_grid1, axis=0) * dx1 / P_E1
E_Amplitud_mod1 = np.sum(beta0_grid[:, 0] * marg_Amplitud_mod1) * dx1
E_sigma_mod1 = np.sum(beta1_grid[0, :] * marg_sigma_mod1) * dy1

print 'Primer modelo: Gaussiana simple'
print 'Amplitud                :', E_Amplitud_mod1
print 'sigma                   :', E_sigma_mod1
print ''


# Modelo 2:
Amplitud1_mod2 = 0.02e-16
sigma1_mod2 = 6.
Amplitud2_mod2 = 0.07e-16
sigma2_mod2 = 1.5
adivinanza_mod2 = [Amplitud1_mod2, 2, sigma1_mod2, 2, Amplitud2_mod2, 2,
                   sigma2_mod2, 2]
beta0_grid, beta1_grid, beta2_grid, beta3_grid = np.mgrid[-0.5:0.5:51j,
                                                          -5:15:51j,
                                                          -0.5:0.5:51j,
                                                          -5:15:51j]
prior_m2, likelihood_m2 = fill_prior_2(beta0_grid, beta1_grid, beta2_grid,
                                       beta3_grid, adivinanza_mod2,
                                       [wavelength, Fnu])

post_grid2 = likelihood_m2 * prior_m2
dx2 = 1 / 200
dy2 = 20 / 200
dz2 = 1 / 200
dt2 = 20 / 200
P_E2 = np.sum(post_grid2) * dx2 * dy2 * dz2 * dt2
marg_Amplitud1_mod2 = (np.sum(np.sum(np.sum(post_grid2, axis=1), axis=1),
                              axis=1) * dy2 * dz2 * dt2 / P_E2)
marg_sigma1_mod2 = (np.sum(np.sum(np.sum(post_grid2, axis=1), axis=1),
                              axis=1) * dx2 * dz2 * dt2 / P_E2)
marg_Amplitud2_mod2 = (np.sum(np.sum(np.sum(post_grid2, axis=1), axis=1),
                              axis=1) * dx2 * dy2 * dt2 / P_E2)
marg_sigma2_mod2 = (np.sum(np.sum(np.sum(post_grid2, axis=1), axis=1),
                              axis=1) * dx2 * dy2 * dz2 / P_E2)

E_Amplitud1_mod2 = np.sum(beta0_grid[:, 0] * marg_Amplitud1_mod2) * dx2
E_sigma1_mod2 = np.sum(beta1_grid[0, :] * marg_sigma1_mod2) * dy2
E_Amplitud2_mod2 = np.sum(beta0_grid[:, 0] * marg_Amplitud2_mod2) * dz2
E_sigma2_mod2 = np.sum(beta1_grid[0, :] * marg_sigma2_mod2) * dt2

print 'Segundo modelo: Gaussiana doble'
print 'Amplitud 1              :', E_Amplitud1_mod2
print 'sigma 1                 :', E_sigma1_mod2
print 'Amplitud 2              :', E_Amplitud2_mod2
print 'sigma 2                 :', E_sigma2_mod2
print ''


# Plots

plt.figure(1)
plt.style.use('bmh')
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plot_distribution(beta0_grid, beta1_grid, prior_m1 * likelihood_m1)
plt.savefig('fig_1.eps')

plt.figure(2)
plt.style.use('bmh')
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.plot(beta0_grid, marg_Amplitud_mod1)
plt.savefig('Densidad_prob_Amod1.eps')

plt.figure(3)
plt.style.use('bmh')
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.plot(beta1_grid, marg_sigma_mod1)
plt.savefig('Densidad_prob_sigmamod1.eps')
plt.show()
