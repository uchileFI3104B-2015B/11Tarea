#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from numpy import sum
import matplotlib.pyplot as plt
from scipy.stats import distributions
from scipy.stats import norm
import pdb
from pymc3 import find_MAP
from pymc3 import Model, Normal, HalfNormal
import theano.tensor as T
from pymc3 import traceplot
from pymc3 import NUTS, sample
from scipy.stats import multivariate_normal

ESCALA = 1e18


def prior(beta, params):
    beta0, beta1, beta2, beta3 = beta
    mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
    S = -1. / 2 * ((beta0 - mu0)**2 / sigma0**2 + (beta1 - mu1)**2 / sigma1**2 + (beta2 - mu2)**2 / sigma2**2 + (beta3 - mu3)**2 / sigma3**2)
    P = np.exp(S) / ((2 * np.pi)**2 * sigma0 * sigma1 * sigma2 * sigma3)
    return P


def likelihood(beta, data, sigma_datos):
    beta0, beta1, beta2, beta3 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    mu = 6563
    S = np.sum((y - 1e-16 * ESCALA - beta0 * norm(loc=mu, scale=beta1).pdf(x) - beta2 * norm(loc=mu, scale=beta3).pdf(x))**2)
    L = (2 * np.pi * sigma_datos**2)**(-N / 2.) * np.exp(-S / 2 / sigma_datos**2)
    return L


def importar_datos(txt):
    ''' función que retorna arreglos por cada columna del archivo de datos'''
    data = np.loadtxt(txt)
    return data[:, 0], data[:, 1] * ESCALA


def calcular_sigma_datos(wl, fnu):
    ''' calcula la dispersión de los datos del archivo en el tramo de la
    constante'''
    MU = 1 * 1e-16 * ESCALA
    i = 0
    sig = 0
    while wl[i] < 6540 or wl[i]>6600:
        sig += (fnu[i] - MU)**2
        i +=1
    return sig


# Main
wl, fnu = importar_datos("espectro.dat")

sigma_datos = calcular_sigma_datos(wl, fnu)
print("sigma datos = ", sigma_datos)

basic_model = Model()

with basic_model:

    # Priors for unknown model parameters
    A = Normal('amplitud', mu=1e-17 * ESCALA, sd=1e16 * ESCALA)
    sigma = Normal('sigma', mu=8, sd=100)
    A2 = Normal('amplitud 2', mu=1e-17 * ESCALA, sd=1e16 * ESCALA)
    sigma2 = Normal('sigma 2', mu=4, sd=100)

    mu = 6563
    # Expected value of outcome
    y_out = 1e-16 * ESCALA - A * 1/(sigma * np.sqrt(2 * np.pi)) * T.exp(-0.5 * ((wl-mu)/sigma)**2) - A2 * 1/(sigma2 * np.sqrt(2 * np.pi)) * T.exp(-0.5 * ((wl-mu)/sigma2)**2)

    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=y_out, sd=sigma_datos, observed=fnu)

map_estimate = find_MAP(model=basic_model)

print(map_estimate)

fig1, ax1 = plt.subplots()
ax1.plot(wl, fnu, label="Datos")
ax1.plot(wl, 1e-16 * ESCALA - float(map_estimate["amplitud"]) * norm(loc=6563, scale=float(map_estimate["sigma"])).pdf(wl) - float(map_estimate["amplitud 2"]) * norm(loc=6563, scale=float(map_estimate["sigma 2"])).pdf(wl), label="fit, gaussiana doble")
ax1.set_title("Datos y fit de Gaussiana")
ax1.set_xlabel("Longitud de onda [Angstrom]")
ax1.set_ylabel("$F_\\nu [erg s^{-1} Hz^{-1}cm^{-2}] \\times 1e17$")
ax1.set_xlim(wl.min(), wl.max())
plt.legend(loc=4)
plt.savefig("fit 2 gaussianas.jpg")

with basic_model:
    trace = sample(500, start=map_estimate)
traceplot(trace)
plt.savefig("graf2.jpg")
plt.show()

# Usaremos una gaussiana 2D
# los parametros adecuados los podemos estimar de la muestra obtenida usando pymc3 o Metropolis.
mu = [map_estimate["amplitud"], map_estimate["sigma"], map_estimate["amplitud 2"], map_estimate["sigma 2"]]
algo = 1e-3
sigma = [[algo,0, 0, 0], [0, algo, 0, 0], [0, 0, algo, 0], [0 ,0, 0, algo]] # Ver que pasa con covarianza = -0.5
nd_normal = multivariate_normal(mu, sigma)

prior_params = [map_estimate["amplitud"], 1e16 * ESCALA, map_estimate["sigma"], 100, map_estimate["amplitud 2"], 1e16 * ESCALA, map_estimate["sigma 2"], 100]
N_sample = 500
random_points = nd_normal.rvs(N_sample)
suma = 0
for p in random_points:
    posterior = likelihood(p, [wl, fnu], sigma_datos) * prior(p, prior_params)
    suma += posterior / nd_normal.pdf(p)
print ("")
print suma / N_sample
