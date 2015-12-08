#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from numpy import sum
import emcee
import matplotlib.pyplot as plt
from scipy.stats import distributions
import scipy.optimize as op
from scipy.stats import norm
import pdb
import corner
''' resuelve la p2 de la tarea, usando MCMC para calcular la
verosimilitud'''

ESCALA = 1e17


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


def lnlike(theta, x, y, yerr):
    '''logaritmo de la verosimilitud'''
    Ndim = 4
    A1, sig1, A2, sig2 = theta
    mu = 6563
    model = 1e-16 * ESCALA - A1 * norm(loc=mu, scale=sig1).pdf(x) + 1e-16 * ESCALA - A2 * norm(loc=mu, scale=sig2).pdf(x)
    inv_sigma2 = yerr
    return -0.5*(np.sum((y-model)**2*inv_sigma2 + Ndim * np.log(inv_sigma2)))


def lnprior(theta, adivinanza):
    '''log prior'''
    Ndim = 4
    A1, sig1, A2, sig2 = theta
    mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3 = adivinanza
    S = -1. / 2 * ((A1 - mu0)**2 / sigma0**2 + (sig1 - mu1)**2 / sigma1**2 + (A2 - mu2)**2 / sigma2**2 + (sig2 - mu3)**2 / sigma3**2)
    lnP = S - 0.5 * np.log((2 * np.pi)**Ndim * sigma0**2 * sigma1**2 * sigma2**2 * sigma3**2)
    return lnP


def lnprob(theta, adiv, x, y, yerr):
    lp = lnprior(theta, adiv)
    return lp + lnlike(theta, x, y, yerr)


wl, fnu = importar_datos("espectro.dat")

sigma_datos = calcular_sigma_datos(wl, fnu)
print("sigma datos = ", sigma_datos)

ndim, nwalkers = 4, 100
adivinanza = [0.06 * 1e-16 * ESCALA, 100, 2, 100, 0.02 * 1e-16 * ESCALA, 100, 10, 100]
pos = [[0.06 * 1e-16 * ESCALA, 2, 0.02 * 1e-16 * ESCALA, 10] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(adivinanza, wl, fnu, sigma_datos))
sampler.run_mcmc(pos, 500)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
                      truths=[m_true, b_true, np.log(f_true)])
fig.savefig("triangle.png")
