#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit
import pymc3 as pm
import theano.tensor as T
from scipy import optimize

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
    A = amplitud ; s = sigma
    '''
    g = A * T.exp(-0.5 * ((x - 6563) / sigma)**2) / (sigma * np.sqrt(2*np.pi))
    return 1e-16 - g


def gauss_simple(x, A, sigma):
    '''
    Retorna la función correspondiente al primer modelo, es decir, una
    gaussiana trasladada en 1e-16
    '''
    return 1e-16 - A * scipy.stats.cauchy(loc=6563, scale=sigma).pdf(x)


def modelo_2(x, A1, s1, A2, s2):
    '''
    Retorna la función correspondiente al primer modelo, es decir, la suma
    de dos gaussianas trasladada en 1e-16.
    A = amplitud ; s = sigma
    '''
    mu = 6563
    g1 = A1 * T.exp(-0.5 * ((x - mu) / s1)**2) / (s1 * np.sqrt(2*np.pi))
    g2 = A2 * T.exp(-0.5 * ((x - mu) / s2)**2) / (s2 * np.sqrt(2*np.pi))
    return 1e-16 - g1 - g2


def gauss_doble(x, Amplitud1, sigma1, Amplitud2, sigma2):
    '''
    Retorna la función correspondiente al primer modelo, es decir, la suma
    de dos gaussianas trasladada en 1e-16
    '''
    gauss1 = (Amplitud1 * scipy.stats.norm(loc=6563, scale=sigma1).pdf(x))
    gauss2 = (Amplitud2 * scipy.stats.norm(loc=6563, scale=sigma2).pdf(x))
    return 1e-16 - gauss1 - gauss2


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
Fnu = np.loadtxt("espectro.dat", usecols=[1]) * 1e+16
x = np.linspace(min(wavelength), max(wavelength), 100)
N = 5000

with pm.Model() as basic_model1:

    # Priors for unknown model parameters
    Amplitud_mod1 = pm.Normal('Amplitud_mod1', mu=0.917, sd=0.08)
    sigma_mod1 = pm.Normal('sigma_mod1', mu=3.7, sd=0.5)

    # Expected value of outcome
    y_out = modelo_1(wavelength, Amplitud_mod1, sigma_mod1)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=y_out, sd=1.5, observed=Fnu)


map_estimate = pm.find_MAP(model=basic_model1, fmin=optimize.fmin_powell)
print(map_estimate)
with basic_model1:
    trace1 = pm.sample(N, start=map_estimate)


# Resultados
E_Amplitud_mod1 = np.mean(trace1.Amplitud_mod1) * 1e-16
E_sigma_mod1 = np.mean(trace1.sigma_mod1)

Amplitud_mod1_sort = np.sort(trace1.Amplitud_mod1) * 1e-16
sigma_mod1_sort = np.sort(trace1.sigma_mod1)

A_limite_bajo = Amplitud_mod1_sort[int(N * 0.16)]
A_limite_alto = Amplitud_mod1_sort[int(N * 0.84)]
s_limite_bajo = sigma_mod1_sort[int(N * 0.16)]
s_limite_alto = sigma_mod1_sort[int(N * 0.84)]

print ''
print 'Primer modelo: Gaussiana simple'
print 'Amplitud                        :', E_Amplitud_mod1
print 'Intervalo de credibilidad al 68%:', [A_limite_bajo, A_limite_alto]
print 'sigma                           :', E_sigma_mod1
print 'Intervalo de credibilidad al 68%:', [s_limite_bajo, s_limite_alto]
print ''


with pm.Model() as basic_model2:
    # Priors for unknown model parameters
    Amplitud1_mod2 = pm.Normal('Amplitud1_mod2', mu=0.6, sd=0.05)
    sigma1_mod2 = pm.Normal('sigma1_mod2', mu=1.5, sd=1.5)
    Amplitud2_mod2 = pm.Normal('Amplitud2_mod2', mu=0.2, sd=0.05)
    sigma2_mod2 = pm.Normal('sigma2_mod2', mu=7, sd=1.5)

    # Expected value of outcome
    y_out = modelo_2(wavelength, Amplitud1_mod2, sigma1_mod2, Amplitud2_mod2,
                     sigma2_mod2)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=y_out, sd=1.5, observed=Fnu)

map_estimate = pm.find_MAP(model=basic_model2, fmin=optimize.fmin_powell)
print(map_estimate)
with basic_model2:
    trace2 = pm.sample(N, start=map_estimate)

# Resultados
E_Amplitud1_mod2 = np.mean(trace2.Amplitud1_mod2) * 1e-16
E_sigma1_mod2 = np.mean(trace2.sigma1_mod2)
E_Amplitud2_mod2 = np.mean(trace2.Amplitud2_mod2) * 1e-16
E_sigma2_mod2 = np.mean(trace2.sigma2_mod2)

Amplitud1_mod2_sort = np.sort(trace2.Amplitud1_mod2) * 1e-16
sigma1_mod2_sort = np.sort(trace2.sigma1_mod2)
Amplitud2_mod2_sort = np.sort(trace2.Amplitud2_mod2) * 1e-16
sigma2_mod2_sort = np.sort(trace2.sigma2_mod2)

A1_limite_bajo = Amplitud1_mod2_sort[int(N * 0.16)]
A1_limite_alto = Amplitud1_mod2_sort[int(N * 0.84)]
s1_limite_bajo = sigma1_mod2_sort[int(N * 0.16)]
s1_limite_alto = sigma1_mod2_sort[int(N * 0.84)]
A2_limite_bajo = Amplitud2_mod2_sort[int(N * 0.16)]
A2_limite_alto = Amplitud2_mod2_sort[int(N * 0.84)]
s2_limite_bajo = sigma2_mod2_sort[int(N * 0.16)]
s2_limite_alto = sigma2_mod2_sort[int(N * 0.84)]

print ''
print 'Segundo modelo: Gaussiana doble'
print 'Amplitud 1                      :', E_Amplitud1_mod2
print 'Intervalo de credibilidad al 68%:', [A1_limite_bajo, A1_limite_alto]
print 'sigma 1                         :', E_sigma2_mod2
print 'Intervalo de credibilidad al 68%:', [s1_limite_bajo, s1_limite_alto]

print 'Amplitud 2                      :', E_Amplitud2_mod2
print 'Intervalo de credibilidad al 68%:', [A2_limite_bajo, A2_limite_alto]
print 'sigma 2                         :', E_sigma2_mod2
print 'Intervalo de credibilidad al 68%:', [s2_limite_bajo, s2_limite_alto]
print ''


plt.figure(1, figsize=(10, 7))
plt.style.use('bmh')
plt.plot(wavelength, Fnu*1e-16, color='brown', drawstyle='steps-post',
         label='Datos')
plt.plot(x, gauss_simple(x, E_Amplitud_mod1, E_sigma_mod1),
         label='Modelo 1 (Gaussiana simple)', linewidth=2.0)
plt.xlim(6520, 6600)
plt.legend(loc='lower left')
plt.xlabel('Wavelength [$\AA$]', fontsize=16)
plt.ylabel('$F_v$[erg s$^{-1}$Hz$^{-1}$cm$^{-2}$]', fontsize=16)
plt.savefig('Fit_mod1.eps')

plt.figure(2)
plt.style.use('bmh')
h, _, _ =  plt.hist(trace1.Amplitud_mod1 * 1e-16,
                    bins=np.arange(0.6e-16, 1.2e-16, 1.5e-18),
                    normed=True, color='g')
plt.axvline(E_Amplitud_mod1, label='Esperanza')
plt.axvline(A_limite_alto, label='Intervalo de credibilidad', color='m')
plt.axvline(A_limite_bajo, color='m')
plt.legend(loc='upper left')
plt.xlabel('Amplitud [erg s$^{-1}$Hz$^{-1}$cm$^{-2}\AA$]')
plt.savefig('Densid_probab_Am1.eps')

plt.figure(3)
plt.style.use('bmh')
h, _, _ = plt.hist(trace1.sigma_mod1, bins=np.arange(2, 5.5, 0.1), normed=True)
plt.axvline(E_sigma_mod1, label='Esperanza')
plt.axvline(s_limite_alto, label='Intervalo de credibilidad', color='m')
plt.axvline(s_limite_bajo, color='m')
plt.xlabel('Varianza [$\AA$]')
plt.legend(loc='upper left')
plt.savefig('Densid_probab_sm1.eps')


plt.figure(4, figsize=(10, 7))
plt.style.use('bmh')
plt.plot(wavelength, Fnu*1e-16, color='brown', drawstyle='steps-post',
         label='Datos')
plt.plot(x, gauss_doble(x, E_Amplitud1_mod2, E_sigma1_mod2, E_Amplitud2_mod2,
                        E_sigma2_mod2), label='Modelo 2 (Gaussiana doble)',
         linewidth=2.0)
plt.plot(x, gauss_simple(x, E_Amplitud1_mod2, E_sigma1_mod2), '--', color='g',
         label='Gaussiana 1 (modelo 2)', alpha=0.8)
plt.plot(x, gauss_simple(x, E_Amplitud2_mod2, E_sigma2_mod2), '-.', color='g',
         label='Gaussiana 2 (modelo 2)', alpha=0.8)
plt.xlim(6520, 6600)
plt.legend(loc='lower left')
plt.xlabel('Wavelength [$\AA$]', fontsize=16)
plt.ylabel('$F_v$[erg s$^{-1}$Hz$^{-1}$cm$^{-2}$]', fontsize=16)
plt.savefig('Fit_mod2.eps')

plt.figure(5)
plt.style.use('bmh')
h, _, _ = plt.hist(trace2.Amplitud1_mod2 * 1e-16,
                   bins=np.arange(4e-17, 8e-17, 1.5e-18),
                   normed=True, color='g')
plt.axvline(E_Amplitud1_mod2, label='Esperanza')
plt.axvline(A1_limite_alto, label='Intervalo de credibilidad', color='m')
plt.axvline(A1_limite_bajo, color='m')
plt.xlabel('Amplitud [erg s$^{-1}$Hz$^{-1}$cm$^{-2}\AA$]')
plt.legend()
plt.savefig('Densid_probab_A1m2.eps')

plt.figure(6)
plt.style.use('bmh')
h, _, _ = plt.hist(trace2.sigma1_mod2, bins=np.arange(-4, 7, 0.3), normed=True)
plt.axvline(E_sigma1_mod2, label='Esperanza')
plt.axvline(s1_limite_alto, label='Intervalo de credibilidad', color='m')
plt.axvline(s1_limite_bajo, color='m')
plt.xlabel('Varianza [$\AA$]')
plt.legend()
plt.savefig('Densid_probab_s1m2.eps')

plt.figure(7)
plt.style.use('bmh')
h, _, _ = plt.hist(trace2.Amplitud2_mod2 * 1e-16,
                   bins=np.arange(0, 4e-17, 1.5e-18),
                   normed=True, color='g')
plt.axvline(E_Amplitud2_mod2, label='Esperanza')
plt.axvline(A2_limite_alto, label='Intervalo de credibilidad', color='m')
plt.axvline(A2_limite_bajo, color='m')
plt.xlabel('Amplitud [erg s$^{-1}$Hz$^{-1}$cm$^{-2}\AA$]')
plt.legend()
plt.savefig('Densid_probab_A2m2.eps')

plt.figure(8)
plt.style.use('bmh')
h, _, _ = plt.hist(trace2.sigma2_mod2, bins=np.arange(2, 12, 0.4), normed=True)
plt.axvline(E_sigma2_mod2, label='Esperanza')
plt.axvline(s2_limite_alto, label='Intervalo de credibilidad', color='m')
plt.axvline(s2_limite_bajo, color='m')
plt.xlabel('Varianza [$\AA$]')
plt.legend()
plt.savefig('Densid_probab_s2m2.eps')

plt.figure(9)
pm.traceplot(trace2)
plt.savefig('plot_diagnostico_mod2')

plt.figure(10)
[[ax11, ax12], [ax21, ax22]] = pm.traceplot(trace1)
ax11.axvline(0.917)
ax21.axvline(3.7)
plt.savefig('plot_diagnostico_mod1')

plt.show()
