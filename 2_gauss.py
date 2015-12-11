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
    A = Normal('amplitud', mu=1e-17 * ESCALA, sd=100)
    sigma = Normal('sigma', mu=8, sd=100)
    A2 = Normal('amplitud 2', mu=1e-17 * ESCALA, sd=100)
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
    trace = sample(5000, start=map_estimate)
traceplot(trace)
plt.savefig("graf2.jpg")
plt.show()
