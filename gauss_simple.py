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
            output[i, j] = prior([beta0_grid[i,j], beta1_grid[i,j]], prior_params)
    return output


def likelihood(beta, data, f, sigma_datos):
    beta0, beta1 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - f(x, beta0, beta1))**2)
    L = (2 * np.pi * sigma_datos**2)**(-N / 2.) * np.exp(-S / 2 / sigma_datos**2)
    return L

def fill_likelihood(beta0_grid, beta1_grid, data, f, sigma_datos):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = likelihood([beta0_grid[i,j], beta1_grid[i,j]], data, f, sigma_datos)

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


def gauss(x, A, sigma):
    '''gaussiana simple para primer modelo'''
    mu = 6563
    return 1e-16 * ESCALA - A * norm(loc=mu, scale=sigma).pdf(x)


def graficos(x, y, b0, b1, post):
    ''' función para graficar llas probabilidades marginales
    de los parámetros y la probabilidad 2D, devuelve las distribuciones
    marginales'''
    marginal_b0 = np.sum(post, axis=1)
    fig2, ax2 = plt.subplots()
    ax2.plot(b0[:, 0], marginal_b0)
    ax2.set_title("Distribucion para la amplitud")
    ax2.set_xlabel("Amplitud $F_\\nu [erg s^{-1} Hz^{-1}cm^{-2}] \\times 1e17$")
    ax2.set_ylabel("Probabilidad")
    plt.savefig("Distribucion para la amplitud.jpg")
    marginal_b1 = np.sum(post, axis=0)

    fig3, ax3 = plt.subplots()
    ax3.plot(b1[0, :], marginal_b1)
    ax3.set_title("Distribucion para la varianza")
    ax3.set_xlabel("varianza $F_\\nu [erg s^{-1} Hz^{-1}cm^{-2}] \\times 1e17$")
    ax3.set_ylabel("Probabilidad")
    plt.savefig("Distribucion para la varianza.jpg")

    fig4, ax4 = plt.subplots()
    ax4.pcolormesh(b0, b1, post)
    ax4.set_xlim(7, 8.4)
    ax4.set_title("Probabilidad post.")
    ax4.set_xlabel("Amplitud")
    ax4.set_ylabel("Varianza")
    plt.savefig("Post.jpg")

    return marginal_b0, marginal_b1

def graficar_fit(x, y, A, sigma, f):
    '''grafica los datos y el fit correspondiente'''
    fig1, ax1 = plt.subplots()
    ax1.plot(x, y, label="Datos")
    ax1.plot(x, f(x, A, sigma), label="fit, gaussiana simple")
    ax1.set_title("Datos y fit de Gaussiana")
    ax1.set_xlabel("Longitud de onda [Angstrom]")
    ax1.set_ylabel("$F_\\nu [erg s^{-1} Hz^{-1}cm^{-2}] \\times 1e17$")
    ax1.set_xlim(x.min(), x.max())
    plt.legend(loc=4)
    plt.savefig("fit.jpg")



def calcular_esperanza_param(b0, marginal_b0, b1,  marginal_b1):
    '''función para calcular la esperanza de los parámetros a partir de
    sus distribuciones marginales'''
    sum_b0 = np.sum(marginal_b0)
    sum_b1 = np.sum(marginal_b1)
    e_b0 = 0
    e_b1 = 0
    for i in range(b0.shape[0]):
        e_b0 += b0[i] * marginal_b0[i]
        e_b1 += b1[i] * marginal_b1[i]
    return e_b0 / sum_b0, e_b1 / sum_b1


# Main
wl, fnu = importar_datos("espectro.dat")

sigma_datos = calcular_sigma_datos(wl, fnu)
print("sigma datos = ", sigma_datos)
beta0_grid, beta1_grid = np.mgrid[7:8.5:99j, 3.4:4:99j]
beta_prior_pars = [1, 100, 5, 100]
prior_grid = fill_prior(beta0_grid, beta1_grid, beta_prior_pars)

likelihood_grid = fill_likelihood(beta0_grid, beta1_grid, [wl, fnu], gauss, sigma_datos)
post_grid = likelihood_grid * prior_grid

marginal_b0, marginal_b1 = graficos(wl, fnu, beta0_grid, beta1_grid, post_grid)
A, sigma = calcular_esperanza_param(beta0_grid[:, 0], marginal_b0, beta1_grid[0], marginal_b1)
print ("valores de esperanza para, amplitud = {}, varianza= {} ".format(A, sigma))
graficar_fit(wl, fnu, A, sigma, gauss)
plt.show()
