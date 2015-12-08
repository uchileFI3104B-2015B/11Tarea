#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from numpy import sum
import matplotlib.pyplot as plt
from scipy.stats import distributions
from scipy.stats import norm
import pdb



''' Script que ajusta una gaussiana a el espectro observado usando métodos
Bayesianos'''

ESCALA = 1e17


def importar_datos(txt):
    ''' función que retorna arreglos por cada columna del archivo de datos'''
    data = np.loadtxt(txt)
    return data[:, 0], data[:, 1] * ESCALA


def prior(beta, params):
    beta0, beta1, beta2, beta3 = beta
    mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
    S = -1. / 2 * ((beta0 - mu0)**2 / sigma0**2 + (beta1 - mu1)**2 / sigma1**2 + (beta2 - mu2)**2 / sigma2**2 + (beta3 - mu3)**2 / sigma3**2)
    P = np.exp(S) / ((2 * np.pi)**2 * sigma0 * sigma1 * sigma2 * sigma3)
    return P


def fill_prior(beta0_grid, beta1_grid, beta2_grid, beta3_grid, prior_params):
    output = np.zeros(beta0_grid.shape)
    ni, nj, nk, nm = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for m in range(nm):
                    output[i, j, k, m] = prior([beta0_grid[i, j, k, m], beta1_grid[i, j, k, m], beta2_grid[i, j, k, m], beta3_grid[i, j, k, m]], prior_params)
    return output


def likelihood(beta, data, f, sigma_datos):
    beta0, beta1, beta2, beta3 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - f(x, beta0, beta1, beta2, beta3))**2)
    L = (2 * np.pi * sigma_datos**2)**(-N / 2.) * np.exp(-S / 2 / sigma_datos**2)
    return L


def fill_likelihood(beta0_grid, beta1_grid, beta2_grid, beta3_grid, data, f, sigma_datos):
    output = np.zeros(beta0_grid.shape)
    ni, nj, nk, nm = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for m in range(nm):
                    output[i, j, k, m] = likelihood([beta0_grid[i, j, k, m], beta1_grid[i, j, k, m], beta2_grid[i, j, k, m], beta3_grid[i, j, k, m]], data, f, sigma_datos)
    return output


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


def gauss(x, A, sigma, A2, sigma2):
    '''gaussiana simple para primer modelo'''
    mu = 6563
    return 1e-16 * ESCALA - A * norm(loc=mu, scale=sigma).pdf(x) + 1e-16 * ESCALA - A2 * norm(loc=mu, scale=sigma2).pdf(x)


def graficos(x, y, b0, b1, b2, b3, post):
    ''' función para graficar llas probabilidades marginales
    de los parámetros y la probabilidad 2D, devuelve las distribuciones
    marginales'''
    marginal_b0 = np.sum(post, axis=(1, 2, 3))
    fig2, ax2 = plt.subplots()
    ax2.plot(b0[:, 0, 0, 0], marginal_b0)
    ax2.set_title("Distribucion para la amplitud")
    ax2.set_xlabel("Amplitud $F_\\nu [erg s^{-1} Hz^{-1}cm^{-2}] \\times 1e17$")
    ax2.set_ylabel("Probabilidad")

    marginal_b1 = np.sum(post, axis=(0, 2, 3))

    fig3, ax3 = plt.subplots()
    ax3.plot(b1[0, :, 0, 0], marginal_b1)
    ax3.set_title("Distribucion para la varianza")
    ax3.set_xlabel("varianza $F_\\nu [erg s^{-1} Hz^{-1}cm^{-2}] \\times 1e17$")
    ax3.set_ylabel("Probabilidad")

    marginal_b2 = np.sum(post, axis=(0, 1, 3))

    fig4, ax4 = plt.subplots()
    ax4.plot(b2[0, 0, :, 0], marginal_b2)
    ax4.set_title("Distribucion para la varianza")
    ax4.set_xlabel("varianza $F_\\nu [erg s^{-1} Hz^{-1}cm^{-2}] \\times 1e17$")
    ax4.set_ylabel("Probabilidad")

    marginal_b3 = np.sum(post, axis=(0, 1, 2))

    fig5, ax5 = plt.subplots()
    ax5.plot(b3[0, 0, 0, :], marginal_b1)
    ax5.set_title("Distribucion para la varianza")
    ax5.set_xlabel("varianza $F_\\nu [erg s^{-1} Hz^{-1}cm^{-2}] \\times 1e17$")
    ax5.set_ylabel("Probabilidad")

    return marginal_b0, marginal_b1, marginal_b2, marginal_b3


def graficar_fit(x, y, A, sigma, A2, sigma2, f):
    '''grafica los datos y el fit correspondiente'''
    fig1, ax1 = plt.subplots()
    ax1.plot(x, y, label="Datos")
    ax1.plot(x, f(x, A, sigma, A2, sigma2), label="fit, gaussiana simple")
    ax1.set_title("Datos y fit de Gaussiana")
    ax1.set_xlabel("Longitud de onda [Angstrom]")
    ax1.set_ylabel("$F_\\nu [erg s^{-1} Hz^{-1}cm^{-2}] \\times 1e17$")
    ax1.set_xlim(x.min(), x.max())
    plt.legend(loc=4)



def calcular_esperanza_param(b0, marginal_b0, b1,  marginal_b1, b2, marginal_b2, b3,  marginal_b3):
    '''función para calcular la esperanza de los parámetros a partir de
    sus distribuciones marginales'''
    sum_b0 = np.sum(marginal_b0)
    sum_b1 = np.sum(marginal_b1)
    sum_b2 = np.sum(marginal_b2)
    sum_b3 = np.sum(marginal_b3)
    e_b0 = 0
    e_b1 = 0
    e_b2 = 0
    e_b3 = 0
    for i in range(b0.shape[0]):
        e_b0 += b0[i] * marginal_b0[i]
        e_b1 += b1[i] * marginal_b1[i]
        e_b2 += b2[i] * marginal_b2[i]
        e_b3 += b3[i] * marginal_b3[i]
    return e_b0 / sum_b0, e_b1 / sum_b1, e_b2 / sum_b2, e_b3 / sum_b3


# Main
wl, fnu = importar_datos("espectro.dat")

sigma_datos = calcular_sigma_datos(wl, fnu)
print("sigma datos = ", sigma_datos)

A1_grid, sig1_grid, A2_grid, sig2_grid = np.mgrid[0.1:1:9j, 0.2:4:9j, 0.1:2:9j, 5:15:9j]

prior_pars = [0.06 * 1e-16 * ESCALA, 100, 2, 100, 0.02 * 1e-16 * ESCALA, 100, 10, 100]
prior_grid = fill_prior(A1_grid, sig1_grid, A2_grid, sig2_grid, prior_pars)

likelihood_grid = fill_likelihood(A1_grid, sig1_grid, A2_grid, sig2_grid, [wl, fnu], gauss, sigma_datos)
post_grid = likelihood_grid * prior_grid

marginal_b0, marginal_b1, marginal_b2, marginal_b3 = graficos(wl, fnu, A1_grid, sig1_grid, A2_grid, sig2_grid, post_grid)
A, sigma, A2, sig2 = calcular_esperanza_param(A1_grid[:, 0, 0, 0], marginal_b0, sig1_grid[0, :, 0, 0], marginal_b1, A2_grid[0, 0, :, 0], marginal_b2, sig2_grid[0, 0, 0, :], marginal_b3)
print ("valores de esperanza para, amplitud_1 = {}, varianza_1= {}, amplitud_2 = {}, varianza_2 = {} ".format(A, sigma, A2, sig2))
graficar_fit(wl, fnu, A, sigma, A2, sig2, gauss)
plt.show()
