#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit

'''
En este código se busca modelar el ensanchamiento de una linea de absorcion
en un espectro de radiación.
Para ello se utilizan dos modelos: uno gaussiano simple y otro con dos
funciones gaussianas
'''
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


def chi2(data, parametros_modelo, funcion_modelo):
    x_datos = data[0]
    y_datos = data[1]
    y_modelo = funcion_modelo(x_datos, *parametros_modelo)
    chi2 = (y_datos - y_modelo)**2
    return np.sum(chi2)


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
adivinanza_mod1 = [Amplitud_mod1, sigma_mod1]
param_optimo1, param_covar1 = curve_fit(modelo_1, wavelength, Fnu,
                                        adivinanza_mod1)
Amplitud_mod1, sigma_mod1 = param_optimo1
chi2_1 = chi2([wavelength, Fnu], param_optimo1, modelo_1)

beta0_grid, beta1_grid = np.mgrid[-4:4:201j, -2:2:201j]
n0, n1 = beta0_grid.shape
prior_m1 = np.zeros((n0, n1))
likelihood_m1 = np.zeros((n0, n1))

for i in range(n0):
    for j in range(n1):
        prior_m1[i, j] = prior([beta0_grid[i, j], beta1_grid[i, j]], [9.9e-17, 0.5, 3.4, 0.5])
        likelihood_m1[i, j] = likelihood([beta0_grid[i, j], beta1_grid[i, j]], [wavelength, Fnu], modelo_1)


print 'Primer modelo: Gaussiana simple'
print 'Amplitud                :', Amplitud_mod1
print 'sigma                   :', sigma_mod1
print 'Chi2                    :', chi2_1
print 'Verosimilitud del modelo:', likelihood([wavelength, Fnu], param_optimo1,
                                              modelo_1)
print ''

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

print 'Segundo modelo: Gaussiana doble'
print 'Amplitud 1              :', Amplitud1_mod2
print 'sigma 1                 :', sigma1_mod2
print 'Amplitud 2              :', Amplitud2_mod2
print 'sigma 2                 :', sigma2_mod2
print 'Chi2                    :', chi2_2
print 'Verosimilitud del modelo:', likelihood([wavelength, Fnu], param_optimo2,
                                              modelo_2)
print ''


# Plots
plt.figure(1)
plt.style.use('bmh')
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plot_distribution(beta0_grid, beta1_grid, prior_m1 * likelihood_m1)


plt.figure(2, figsize=(10, 7))
plt.style.use('bmh')
plt.plot(wavelength, Fnu, color='brown', drawstyle='steps-post',
         label='Datos')
plt.plot(x, modelo_1(x, *param_optimo1), label='Modelo 1 (Gaussiana simple)',
         linewidth=2.0)
plt.plot(x, modelo_2(x, *param_optimo2), '--', color='fuchsia',  linewidth=2.0,
         label='Modelo 2 (Gaussiana doble)')
plt.plot(x, modelo_1(x, Amplitud1_mod2, sigma1_mod2), '--', color='g',
         label='Gaussiana 1 (modelo 2)', alpha=0.8)
plt.plot(x, modelo_1(x, Amplitud2_mod2, sigma2_mod2), '-.', color='g',
         label='Gaussiana (modelo 2)', alpha=0.8)
plt.xlabel('Wavelength [$\AA$]', fontsize=16)
plt.ylabel('$F_v$[erg s$^{-1}$Hz$^{-1}$cm$^{-2}$]', fontsize=16)
plt.xlim(6520, 6600)
plt.ylim(0.9e-16, 1.02e-16)
plt.legend(loc='lower left')
plt.grid(False)
# plt.savefig('Fits_espectro.eps')


plt.show()
