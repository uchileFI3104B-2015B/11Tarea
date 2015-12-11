#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit
import pymc3 as pm
import theano.tensor as T

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
    ax_marginal_x.plot(x[:,0], marginal_x)
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


# Main
# Datos
wavelength = np.loadtxt("espectro.dat", usecols=[0])
Fnu = np.loadtxt("espectro.dat", usecols=[1])
x = np.linspace(min(wavelength), max(wavelength), 100)

# Modelo 1:
Amplitud_mod1 = Fnu.max() - Fnu.min()
sigma_mod1 = 6  # Del gráfico
adivinanza_mod1 = [Amplitud_mod1, 0.1e-16, sigma_mod1, 2]


beta0_grid, beta1_grid = np.mgrid[-4:4:201j, -2:2:201j]
n0, n1 = beta0_grid.shape
prior_m1 = np.zeros((n0, n1))
likelihood_m1 = np.zeros((n0, n1))
for i in range(n0):
    for j in range(n1):
        prior_m1[i, j] = prior([beta0_grid[i, j], beta1_grid[i, j]], adivinanza_mod1)
        likelihood_m1[i, j] = likelihood([beta0_grid[i, j], beta1_grid[i, j]], [wavelength, Fnu], modelo_1)

with pm.Model() as basic_model:

    # Priors for unknown model parameters
    Amplitud_mod1 = pm.Normal('Amplitud_mod1', mu=adivinanza_mod1[0], sd=adivinanza_mod1[1])
    sigma_mod1 = pm.Normal('sigma_mod1', mu=adivinanza_mod1[2], sd=adivinanza_mod1[3])

    # Expected value of outcome
    y_out = Amplitud_mod1 * (1 / (sigma_mod1* np.sqrt(2*np.pi))) * T.exp (-0.5 * ((wavelength - 6563) / sigma_mod1)**2)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=y_out, sd=1.5, observed=Fnu)


map_estimate = pm.find_MAP(model=basic_model)

print(map_estimate)
with basic_model:
    trace = pm.sample(5000, start=map_estimate)



#
# print 'Primer modelo: Gaussiana simple'
# print 'Amplitud                :', Amplitud_mod1
# print 'sigma                   :', sigma_mod1
# print 'Chi2                    :', chi2_1
# print 'Verosimilitud del modelo:', likelihood([wavelength, Fnu], param_optimo1,
#                                               modelo_1)
# print ''

# Modelo 2:
Amplitud1_mod2 = 0.02e-16
sigma1_mod2 = 6.
Amplitud2_mod2 = 0.07e-16
sigma2_mod2 = 1.5
adivinanza_mod2 = [Amplitud1_mod2, sigma1_mod2, Amplitud2_mod2, sigma2_mod2]
param_optimo2, param_covar2 = curve_fit(modelo_2, wavelength, Fnu,
                                        adivinanza_mod2)
Amplitud1_mod2, sigma1_mod2, Amplitud2_mod2, sigma2_mod2 = param_optimo2
chi2_2 = chi2([wavelength, Fnu], param_optimo2, modelo_2)

# print 'Segundo modelo: Gaussiana doble'
# print 'Amplitud 1              :', Amplitud1_mod2
# print 'sigma 1                 :', sigma1_mod2
# print 'Amplitud 2              :', Amplitud2_mod2
# print 'sigma 2                 :', sigma2_mod2
# print 'Chi2                    :', chi2_2
# print 'Verosimilitud del modelo:', likelihood([wavelength, Fnu], param_optimo2,
#                                               modelo_2)
# print ''


# Plots
plt.figure(1)
plt.style.use('bmh')
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plot_distribution(beta0_grid, beta1_grid, prior_m1 * likelihood_m1)

plt.figure(2)
plt.pcolormesh(beta0_grid, beta1_grid, likelihood_m1 * prior_m1, cmap='PuBu_r')
plt.xlim(-4, 4)
plt.ylim(-2,  2)
plt.plot(trace.beta0, trace.beta1, marker='None', ls='-', lw=0.3, color='w')

plt.show()
