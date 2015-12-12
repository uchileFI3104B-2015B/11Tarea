#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############################################################################
#                                TAREA 11                                   #
#############################################################################

'''
Universidad de Chile
Facultad de Ciencias Fisicas y Matematicas
Departamento de Fisica
FI3104 Metodos Numericos para la Ciencia y la Ingenieria
Semestre Primavera 2015

Nicolas Troncoso Kurtovic
'''

from __future__ import division
import matplotlib.pyplot as p
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
import os

#############################################################################
#                                                                           #
#############################################################################


def llamar_archivo(nombre):
    '''
    Llama un archivo 'nombre.dat' de dos columnas, este archivo esta
    ubicado en una carpeta 'data' que se encuentra en el mismo directorio
    que el codigo.
    '''
    cur_dir = os.getcwd()
    arch = np.loadtxt(os.path.join(cur_dir, nombre))
    x = arch[:, 0]
    y = arch[:, 1]
    return x, y


def doblegauss(x, A, sigma1, B, sigma2):
    '''
    Retorna el valor de la suma de dos gaussianas evaluadas en x.
    '''
    g1 = (A * scipy.stats.norm(loc=6563, scale=sigma1).pdf(x))
    g2 = (B * scipy.stats.norm(loc=6563, scale=sigma2).pdf(x))
    return 1e-16 - g1 - g2


def Gauss2D(x, y, A, mu, sigma_pars):
    '''
    Version mas general de doblegauss, considera la covarianza entre ambas
    gaussianas, pero ambas tienen la misma amplitud. Es una gaussiana en 2D.
    '''
    mu_x, mu_y = mu
    sigma_x, sigma_y, rho = sigma_pars
    B = A / (2. * np.pi * sigma_x * sigma_y * np.sqrt(1-rho**2))
    C = -1. / (2. * (1. - rho**2))
    D = (x - mu_x)**2 / sigma_x**2
    E = (y - mu_y)**2 / sigma_y**2
    F = 2. * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y)
    return B * np.exp(C * (D + E - F))


def gauss(x, A, sigma):
    '''
    Retorna el valor de la suma de dos gaussianas evaluadas en x.
    '''
    g = (A * scipy.stats.norm(loc=6563, scale=sigma).pdf(x))
    return 1e-16 - g


def prior(beta, params, dim):
    '''
    Probabilidad de obtener beta a partir de los parametros params. dim
    indica el numero de dimensiones, que puede ser 2D o 4D.
    '''
    if dim == 4:
        beta0, beta1, beta2, beta3 = beta
        mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
        S0 = (beta0 - mu0)**2 / sigma0**2
        S1 = (beta1 - mu1)**2 / sigma1**2
        S2 = (beta2 - mu2)**2 / sigma2**2
        S3 = (beta3 - mu3)**2 / sigma3**2
        S = -1. / 2 * (S0 + S1 + S2 + S3)
        msigmas = sigma0 * sigma1 * sigma2 * sigma3
        P = np.exp(S) / (4. * np.pi**2 * msigmas)
    if dim == 2:
        beta0, beta1 = beta
        mu0, sigma0, mu1, sigma1 = params
        S0 = (beta0 - mu0)**2 / sigma0**2
        S1 = (beta1 - mu1)**2 / sigma1**2
        S = -1. / 2 * (S0 + S1)
        msigmas = sigma0 * sigma1
        P = np.exp(S) / (2 * np.pi * msigmas)
    else:
        return False
    return P


def fill_prior(b_grid, params, dim):
    '''
    Crea una malla de Probabilidades para las combinaciones de variables.
    Los input son las mallas de parametros, mas los parametros que modelan
    la gaussiana de distribucion de los parametros buscados.
    '''
    if dim == 2:
        b0_grid, b1_grid = b_grid
        ni, nj = b0_grid.shape
        output = np.zeros([ni, nj])
        for i in range(ni):
            for j in range(nj):
                coord = [b0_grid[i, j], b1_grid[i, j]]
                output[i, j] = prior(coord, params, dim)
    if dim == 4:
        b0_grid, b1_grid, b2_grid, b3_grid = b_grid
        ni, nj, nk, nl = b0_grid.shape
        output = np.zeros([ni, nj, nk, nl])
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        coord = [b0_grid[i, j], b1_grid[i, j],
                                 b2_grid[i, j], b3_grid[i, j]]
                        output[i, j, k, l] = prior(coord, params, dim)
    return output


def likelihood(beta, x, y, error, dim):
    '''
    Calcula la verosimilitud segun los parametros ingresados en beta,
    utilizando las mediciones 'x, y' que poseen un error std 'error'. Debe
    especificarse el numero de dimensiones.
    '''
    if dim == 2:
        A, sigma = beta
        try:
            N = len(x)
        except:
            N = 1
        S = np.sum((y - gauss(x, A, sigma))**2)
        L = (2 * np.pi * error**2)**(-N / 2.) * np.exp(-S / (2 * error**2))
    if dim == 4:
        A, sigma1, B, sigma2 = beta
        try:
            N = len(x)
        except:
            N = 1
        S = np.sum((y - doblegauss(x, A, sigma1, B, sigma2))**2)
        L = (2 * np.pi * error**2)**(-N / 2.) * np.exp(-S / (2 * error**2))
    return L


def fill_likelihood(b_grid, x, y, error, dim):
    '''
    Crea una malla de Verosimilitudes para las combinaciones de variables.
    Los input son las mallas de parametros, mas los parametros que modelan
    la gaussiana de distribucion de los parametros buscados.
    '''
    if dim == 2:
        b0_grid, b1_grid = b_grid
        ni, nj = b0_grid.shape
        output = np.zeros([ni, nj])
        for i in range(ni):
            for j in range(nj):
                coor = [b0_grid[i, j], b1_grid[i, j]]
                output[i, j] = likelihood(coor, x, y, error, dim)
    if dim == 4:
        b0_grid, b1_grid, b2_grid, b3_grid = b_grid
        ni, nj, nk, nl = b0_grid.shape
        output = np.zeros([ni, nj, nk, nl])
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        coord = [b0_grid[i, j], b1_grid[i, j],
                                 b2_grid[i, j], b3_grid[i, j]]
                        output[i, j, k, l] = likelihood(coor, x, y, error, dim)
    return output


def chi(func, x, y, pars):
    chi2 = np.sum((y - func(x, *pars))**2)
    return chi2


def fitear_implementado(func, x, y, seed=False):
    '''
    Retorna los coeficientes b y a correspondientes a la ecuacion lineal que
    mejor fitea los vectores x e y segun curve_fit.
    '''
    if seed is False:
        popt, pcov = scipy.optimize.curve_fit(func, x, y)
        return popt
    else:
        popt, pcov = scipy.optimize.curve_fit(func, x, y, p0=seed)
        return popt


def desviacion(y, imin, imax):
    '''
    Corta el arreglo x en los indices imin y imax, luego concatena los dos
    arreglos resultantes y calcula el error std. Para realizar esto se
    utilizan listas
    '''
    a = list(y[:imin])
    b = list(y[imax:])
    a.extend(b)
    z = np.array(a)
    return np.std(z)


def resultados_gauss(x, y, seeds=False):
    '''
    Imprime y grafica los resultados del ajuste doble gaussiano a los
    datos x, y
    '''
    # Ajustar gaussiana
    popt1 = fitear_implementado(gauss, x, y, seed=[seeds[0], seeds[1]])
    # Escribir resultados
    print '----------------------------------------------'
    print 'f(x) =  1e-16 - A * N(x, 6563, sigma1)'
    print 'A = ', popt1[0]
    print 'sigma1 = ', popt1[1]
    print 'chi**2 = ', chi(gauss, x, y, popt1)
    # Ajustar doble gaussiana
    popt2 = fitear_implementado(doblegauss, x, y, seed=seeds)
    # Escribir resultados
    print '----------------------------------------------'
    print 'f(x) =  1e-16 - A * N(x, 6563, sigma1) - B * N(x, 6563, sigma2)'
    print 'A = ', popt2[0]
    print 'B = ', popt2[2]
    print 'sigma1 = ', popt2[1]
    print 'sigma2 = ', popt2[3]
    print 'chi**2 = ', chi(doblegauss, x, y, popt2)
    print '----------------------------------------------'
    # Generar datos para ploteo
    X = np.linspace(np.min(x), np.max(x), 10**5)
    # Graficar doble gaussiana
    p.plot(x, y, color='turquoise', drawstyle='steps-post', lw=2., alpha=0.8)
    p.plot(X, doblegauss(X, *popt2), 'b', lw=2., alpha=0.8)
    p.plot(X, gauss(X, popt2[0], popt2[1]), 'g--', lw=2., alpha=0.8)
    p.plot(X, gauss(X, popt2[2], popt2[3]), 'g--', lw=2., alpha=0.8)
    p.axis([6503, 6623, 9e-17, 1.01e-16])
    p.xlabel('Angstrom')
    p.ylabel('erg / s / Hz / cm^2')
    p.grid(True)
    p.show()
    # Graficar gaussiana
    p.plot(x, y, color='turquoise', drawstyle='steps-post', lw=2., alpha=0.8)
    p.plot(X, gauss(X, popt1[0], popt1[1]), 'b', lw=2., alpha=0.8)
    p.axis([6503, 6623, 9e-17, 1.01e-16])
    p.xlabel('Angstrom')
    p.ylabel('erg / s / Hz / cm^2')
    p.grid(True)
    p.show()


#############################################################################
#                               AJUSTE DATOS                                #
#############################################################################

# Leer datos
x, y = llamar_archivo('espectro.dat')

# Puntos escogidos como inicio de la linea de absorcion
p.plot(x[50], y[50], '*')
p.plot(x[76], y[76], '*')
p.plot(x, y, color='turquoise', drawstyle='steps-post', lw=2., alpha=0.8)
p.axis([6503, 6623, 9e-17, 1.01e-16])
p.show()

# Semillas para el ajuste
seeds = [1e-17, 7., 1e-17, 7.]

# Resultados
resultados_gauss(x, y, seeds)
error = desviacion(y, 50, 76)
print 'Desviacion STD fuera de la linea de absorcion = ', error

#############################################################################
#                             DOS DIMENSIONES                               #
#############################################################################

b_grid = np.mgrid[4e-17:11e-17:30j, 1:6:30j]
prior_params = [7.6e-17, 1e-17, 3.7, 1]

#b_grid = np.mgrid[4:11:30j, 1:6:30j]
#prior_params = [7.6, 1, 3.7, 1]


prior_grid = fill_prior(b_grid, prior_params, 2)
b0_grid, b1_grid = b_grid

likelihood_grid = fill_likelihood(b_grid, x, y, 1, 2)
post_grid = likelihood_grid * prior_grid


#'''
p.pcolormesh(b0_grid * 10**17, b1_grid, prior_grid)
p.gca().set_aspect('equal')

p.show()

#'''

