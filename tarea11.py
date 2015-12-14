from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit


def datos():
    archivo = np.loadtxt('espectro.dat')
    wave = archivo[:, 0]
    flux = archivo[:, 1]
    return wave, flux


def gauss(x, A, sigma):
    '''
    Gaussiana simple
    '''
    mu = 6563
    g = (A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x))
    return 1e-16 - g


def doblegauss(x, A1, sigma1, A2, sigma2):
    '''
    Suma de las dos gaussianas
    '''
    mu = 6563
    g1 = A1 * scipy.stats.norm(loc=mu, scale=sigma1).pdf(x)
    g2 = A2 * scipy.stats.norm(loc=mu, scale=sigma2).pdf(x)
    return 1e-16 - g1 - g2


def fit(funcion, x, y, seeds):
    '''
    Calcula coeficientes de la ecuacion que
    mejor fitea x,y.
    '''
    popt, pcov = curve_fit(funcion, x, y, p0=seeds)
    return popt


def chi(funcion, x, y, seeds_op):
    '''
    Retorna el chi^2 asociado a la ecuacion entrante
    '''
    X = 0
    n = len(x)
    for i in range(n):
        X += (y[i] - funcion(x[i], *seeds_op))**2
    return X


# Funciones Profe
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
            output[i, j] = prior([beta0_grid[i, j], beta1_grid[i, j]],
                                 prior_params)
    return output


def likelihood(beta, data):
    beta0, beta1 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - beta0 - beta1 * x)**2)
    L = (2 * np.pi * 1.5**2)**(-N / 2.) * np.exp(-S / 2 / 1.5**2)
    return L


def fill_likelihood(beta0_grid, beta1_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = likelihood([beta0_grid[i, j], beta1_grid[i, j]],
                                      data)
    return output

# - - - # - - - # - - - # - - - #
# Setup
x, y = datos()
X = np.linspace(np.min(x), np.max(x), 10**4)
seeds = 1e-17, 10., 1e-17, 10.  # semillas

# Main
popt1 = fit(gauss, x, y, seeds=[seeds[0], seeds[1]])  # Ajustar gaussiana
popt2 = fit(doblegauss, x, y, seeds=seeds)  # Ajustar doble gaussiana

# Escribir resultados
print 'Parametros 1Gauss:\nA = ', popt1[0], '\nsigma1 = ', popt1[1],\
    '\nX**2 = ', chi(gauss, x, y, popt1)
# Escribir resultados
print 'Parametros 2Gauss:\nA1 = ', popt2[0], '\nA2 = ', popt2[2],\
    '\nsigma1 = ', popt2[1], '\nsigma2 = ', popt2[3], '\nX**2 = ',\
    chi(doblegauss, x, y, popt2)
# Generar datos para ploteo

# gaussiana
plt.plot(x, y, 'go', alpha=0.8, label='Data')
plt.plot(X, gauss(X, popt1[0], popt1[1]), 'r', lw=1.5, label='Gaussiana')
plt.axis([6460, 6660, 9e-17, 1.01e-16])
plt.xlabel('Wave $[Angstroms]$')
plt.ylabel('Flux $[erg / s Hz cm^2]$')
plt.legend(loc='lower right')
plt.show()

# doble gaussiana
plt.plot(x, y, 'go', alpha=0.8, label='Data')
plt.plot(X, doblegauss(X, *popt2), 'r', lw=1.5, alpha=0.8,
         label='Suma Gaussianas')
plt.plot(X, gauss(X, popt2[0], popt2[1]), 'y--', label='Gaussiana 1')
plt.plot(X, gauss(X, popt2[2], popt2[3]), 'y--', label='Gaussiana 2')
plt.axis([6460, 6660, 9e-17, 1.01e-16])
plt.xlabel('$[Angstroms]$')
plt.ylabel('Flux $[erg/ s  Hz  cm^2]$')
plt.legend(loc='lower right')
plt.show()
