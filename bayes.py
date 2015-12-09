from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def leer_archivo(nombre):
    '''
    lee el archivo
    nombre debe ser un str
    '''
    datos = np.loadtxt(nombre)
    x = datos[:, 0]
    y = datos[:, 1]
    return x, y


def modelo_1(p, x):
    A, sigma = p
    a = 10 ** (- 16)
    mu = 6563
    y = a - A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return y


def modelo_2(p, x):
    A1, sigma1, A2, sigma2 = p
    a = 10 ** (- 16)
    mu = 6563
    y = a - A1 * scipy.stats.norm(loc=mu, scale=sigma1).pdf(x) - A2 * scipy.stats.norm(loc=mu, scale=sigma2).pdf(x)
    return y


def gauss2d(x, y, mat_sigma, ):
    sigma_x, sigma_y, rho = mat_sigma
    A = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))
    B = (- 1 / (2 * (2 - rho ** 2))) * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2
                                        - 2 * rho * x * y / (sigma_x * sigma_y))
    return A * np.exp(B)


def prior(beta, p, model=2):
    if model == 2:
        beta0, beta1, beta2, beta3 = beta
        a0, a1, a2, a3, b0, b1, b2, b3 = p
        s0 = ((beta0 - a0) / b0) ** 2
        s1 = ((beta1 - a1) / b1) ** 2
        s2 = ((beta2 - a2) / b2) ** 2
        s3 = ((beta3 - a3) / b3) ** 2
        s = (s0 + s1 + s2 + s3) / 2.
        P = np.exp(s) / (4 * np.pi ** 2 * b0 * b1 * b2 * b3)
        return P
    elif model == 1:
        beta0, beta1 = beta
        a0, a1, b0, b1 = p
        s0 = ((beta0 - a0) / b0) ** 2
        s1 = ((beta1 - a1) / b1) ** 2
        s = (s0 + s1) / 2.
        P = np.exp(s) / (2 * np.pi * b0 * b1)
        return P


def fill_prior(beta_grid, prior_p, model=2):
    if model == 2:
        beta0_grid, beta1_grid, beta2_grid, beta3_grid = beta_grid
        salida = np.zeros(beta0_grid)
        ni, nj, nk, nl = beta0_grid.shape
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        salida[i, j, k, l] = prior([beta0_grid[i, j, k, l], beta1_grid[i,j, k, l], beta2_grid[i, j, k, l], beta3_grid[i,j, k, l]], prior_p, modelo=2)
    elif model == 1:
        beta0_grid, beta1_grid = beta_grid
        salida = np.zeros(beta0_grid)
        ni, nj = beta0_grid.shape
        for i in range(ni):
            for j in range(nj):
                salida[i, j] = prior([beta0_grid[i, j], beta1_grid[i,j]], prior_p, 1)
    return salida


def likelihood(beta, datos, error, model=2):
    x, y = datos
    try:
        N = len(x)
    except:
        N = 1
    if model == 2:
        s = np.sum(y - modelo_2(beta, x))
        L = (2 * np.pi * error ** 2) ** (-N / 2.) * np.exp(- s / (2 * error ** 2))
    elif model == 1:
        s = np.sum(y - modelo_1(beta, x))
        L = (4 * np.pi ** 2 * error ** 4) ** (-N / 2) * np.exp(-s / (2 * error ** 2))
    return L


def fill_likelihood(beta_grid, datos, error, model=2):
    if model == 2:
        beta0_grid, beta1_grid, beta2_grid, beta3_grid = beta_grid
        salida = np.zeros(beta0_grid)
        ni, nj, nk, nl = beta0_grid.shape
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        salida[i, j, k, l] = likelihood([beta0_grid[i, j, k, l], beta1_grid[i,j, k, l], beta2_grid[i, j, k, l], beta3_grid[i,j, k, l]], prior_p, modelo=2)
    elif model == 1:
        beta0_grid, beta1_grid = beta_grid
        salida = np.zeros(beta0_grid)
        ni, nj = beta0_grid.shape
        for i in range(ni):
            for j in range(nj):
                salida[i, j] = likelihood([beta0_grid[i, j], beta1_grid[i,j]], prior_p, 1)
    return salida


def chi_cuad(f, x, y, p):
    n = len(x)
    chi = 0
    for i in range(n):
        chi = chi + (y[i] - f(x[i], p))
    return chi
