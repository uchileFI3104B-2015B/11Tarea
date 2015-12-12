import numpy as np
import matplotlib.pyplot as plt


# Funciones a ocupar
def gauss(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2*sigma**2))/np.sqrt(2*np.pi*sigma**2)


def prior_1(beta, params):
    A, std = beta
    mu0, sigma0, mu1, sigma1 = params
    S = -1. / 2 * ((beta0 - mu0)**2 / sigma0**2 + (beta1 - mu1)**2 / sigma1**2)
    P = np.exp(S) / (2 * np.pi * sigma0 * sigma1)
    return P


def prior_2(beta, params):
    A1, std1, A2, std2 = beta
    mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
    S = -1/2.0 * ((Amp1 - mu0)**2 / sigma0**2 + (std1 - mu1)**2 / sigma1**2 +
                  (Amp2 - mu2)**2 / sigma2**2 + (std2 - mu3)**2 / sigma3**2)
    P = np.exp(S) / ((2*np.pi)**2 * sigma0 * sigma1 * sigma2 * sigma3)
    return P


def fill_prior_1(beta0_grid, beta1_grid, prior_params):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = prior_1([beta0_grid[i,j], beta1_grid[i,j]],
                                    prior_params)
    return output


def fill_prior_2(beta0_grid, beta1_grid, beta2_grid, beta3_grid, prior_params):
    output = np.zeros(par0_grid.shape)
    ni, nj, nk, nl = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for l in range(nl):
                    out[i, j, k, l] = prior_2([par0_grid[i, j, k, l],
                                               par1_grid[i, j, k, l],
                                               par2_grid[i, j, k, l],
                                               par3_grid[i, j, k, l]],
                                               prior_pars)
    return out


def likelihood_1(beta, data):
    beta0, beta1 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - beta0 - beta1 * x)**2)
    L = (2 * np.pi * ruido**2)**(-N / 2.) * np.exp(-S / 2 / ruido**2)
    return L


def likelihood_2(beta, data):
    beta0, beta1, beta2, beta3 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y + beta0 * gauss(x, 6563, beta1) +
               beta2 * gauss(x, 6563, beta3))**2)
    L = (2 * np.pi * ruido**2)**(-N / 2.) * np.exp(-S / 2 / ruido**2)
    return L


def fill_likelihood_1(beta0_grid, beta1_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = likelihood([beta0_grid[i,j], beta1_grid[i,j]], data)
    return output


def fill_likelihood_2(beta0_grid, beta1_grid, beta2_grid, beta3_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj, nk, nl = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for l in range(nl):
                    output[i, j, k, l] = likelihood2([beta0_grid[i, j, k, l],
                                                      beta1_grid[i, j, k, l],
                                                      beta2_grid[i, j, k, l],
                                                      beta3_grid[i, j, k, l]],
                                                      data)
    return output

# Datos
wavelength = np.loadtxt("espectro.dat", usecols=(0,))
fnu = np.loadtxt("espectro.dat", usecols=(1,))
