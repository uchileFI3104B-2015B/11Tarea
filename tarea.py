from __future__ import division
import numpy as np
import matplotlib.pyplot as plt



# Funciones a utilizar (analogas al DEMO)

def gauss(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2*sigma**2))/np.sqrt(2*np.pi*sigma**2)


# Densidad a priori 1
def prior_1(beta, params):
    A, std = beta
    mu0, sigma0, mu1, sigma1 = params
    S = -1. / 2 * ((A - mu0)**2 / sigma0**2 + (std - mu1)**2 / sigma1**2)
    P = np.exp(S) / (2 * np.pi * sigma0 * sigma1)
    return P


# Densidad a priori 2
def prior_2(beta, params):
    A1, std1, A2, std2 = beta
    mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
    S = -1/2.0 * ((A1 - mu0)**2 / sigma0**2 + (std1 - mu1)**2 / sigma1**2 +
                  (A2 - mu2)**2 / sigma2**2 + (std2 - mu3)**2 / sigma3**2)
    P = np.exp(S) / ((2*np.pi)**2 * sigma0 * sigma1 * sigma2 * sigma3)
    return P


# Probabilidad a priori 1
def fill_prior_1(beta0_grid, beta1_grid, prior_params):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = prior_1([beta0_grid[i,j], beta1_grid[i,j]],
                                    prior_params)
    return output


# Probabilidad a priori 2
def fill_prior_2(beta0_grid, beta1_grid, beta2_grid, beta3_grid, prior_params):
    output = np.zeros(beta0_grid.shape)
    ni, nj, nk, nl = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for l in range(nl):
                    output[i, j, k, l] = prior_2([beta0_grid[i, j, k, l],
                                               beta1_grid[i, j, k, l],
                                               beta2_grid[i, j, k, l],
                                               beta3_grid[i, j, k, l]],
                                               prior_params)
    return output


# Similitud 1 para un set de parametros
def likelihood_1(beta, data):
    beta0, beta1 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - 1 + beta0 * gauss(x, 6563, beta1))**2)
    L = (2 * np.pi * ruido**2)**(-N / 2.) * np.exp(-S / (2. * ruido**2))
    return L


# Similitud 2 para un set de parametros
def likelihood_2(beta, data):
    beta0, beta1, beta2, beta3 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - 1 + beta0 * gauss(x, 6563, beta1) +
               beta2 * gauss(x, 6563, beta3))**2)
    L = (2 * np.pi * ruido**2)**(-N / 2.) * np.exp(-S / 2 / ruido**2)
    return L


# Similitud 1 para todo parametros
def fill_likelihood_1(beta0_grid, beta1_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = likelihood_1([beta0_grid[i,j], beta1_grid[i,j]], data)
    return output


# Similitud 2 para todo parametros
def fill_likelihood_2(beta0_grid, beta1_grid, beta2_grid, beta3_grid, data):
    output = np.zeros(beta0_grid.shape)
    ni, nj, nk, nl = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for l in range(nl):
                    output[i, j, k, l] = likelihood_2([beta0_grid[i, j, k, l],
                                                      beta1_grid[i, j, k, l],
                                                      beta2_grid[i, j, k, l],
                                                      beta3_grid[i, j, k, l]],
                                                      data)
    return output


# importacion datos
wavelength = np.loadtxt("espectro.dat", usecols=(0,))
fnu = np.loadtxt("espectro.dat", usecols=(1,)) * (10**16)

# determinacion ruido
dif_cuadrado = 0
n = 0
for i in range(len(fnu)):
    if wavelength[i] <= 6530 or wavelength[i] >= 6600:  # rango continuo
        dif_cuadrado += (fnu[i] - 1)**2
        n += 1
ruido = np.sqrt(dif_cuadrado/n)

'''
-----------------------------------------------------------------------------
Modelo 1:
'''
A1 = fnu.max() - fnu.min()
sigma1 = 6  # Del grafico
adivinanza1 = [A1, 1, sigma1, 3]


beta0_grid1, beta1_grid1 = np.mgrid[0.73:0.8:201j, 3.4:4:201j]
prior_m1 = fill_prior_1(beta0_grid1, beta1_grid1, adivinanza1)
likelihood_m1 = fill_likelihood_1(beta0_grid1, beta1_grid1,
                                   [wavelength, fnu])
post_grid1 = likelihood_m1 * prior_m1

dx1 = 10. / 200
dy1 = 2. / 200
P_E1 = np.sum(post_grid1) * dx1 * dy1
marg_A1 = np.sum(post_grid1, axis=1) * dy1 / P_E1
marg_sigma1 = np.sum(post_grid1, axis=0) * dx1 / P_E1
E_A1 = np.sum(beta0_grid1[:, 0] * marg_A1) * dx1
E_sigma1 = np.sum(beta1_grid1[0, :] * marg_sigma1) * dy1

print 'Primer modelo: Gaussiana simple'
print 'Amplitud                :', E_A1
print 'sigma                   :', E_sigma1
print ''

'''
-----------------------------------------------------------------------------
Modelo 2:
'''

adivinanza2 =[9., 4., 2.5, 1., 10., 5., 8.5, 3.]
beta0_grid2, beta1_grid2, beta2_grid2, beta3_grid2 = np.mgrid[0.34:0.5:51j,
                                                              2:3:51j,
                                                              0.38:0.6:51j,
                                                              6:11:51j]
prior_grid2 = fill_prior_2(beta0_grid2, beta1_grid2, beta2_grid2,
                          beta3_grid2, adivinanza2)
likelihood_grid2 = fill_likelihood_2(beta0_grid2, beta1_grid2, beta2_grid2,
                                    beta3_grid2, [wavelength, fnu])
post_grid2 = likelihood_grid2 * prior_grid2
dx2 = 1.0 / 50
dy2 = 3.0 / 50
dj2 = 1.0 / 50
dk2 = 6.0 / 50
P_E2 = np.sum(post_grid2) * dx2 * dy2 * dj2 * dk2
marg_A2_1 = (np.sum(np.sum(np.sum(post_grid2, axis=1), axis=1), axis=1) *
              dy2 * dj2 * dk2 / P_E2)
marg_sigma2_1 = (np.sum(np.sum(np.sum(post_grid2, axis=0), axis=1), axis=1) *
              dx2 * dj2 * dk2 / P_E2)
marg_A2_2 = (np.sum(np.sum(np.sum(post_grid2, axis=0), axis=0), axis=1) *
              dx2 * dy2 * dk2 / P_E2)
marg_sigma2_2 = (np.sum(np.sum(np.sum(post_grid2, axis=0), axis=0), axis=0) *
              dx2 * dy2 * dj2 / P_E2)
E_A2_1 = np.sum(beta0_grid2[:, 0, 0, 0] * marg_A2_1) * dx2
E_sigma2_1 = np.sum(beta1_grid2[0, :, 0, 0] * marg_sigma2_1) * dy2
E_A2_2 = np.sum(beta2_grid2[0, 0, :, 0] * marg_A2_2) * dj2
E_sigma2_2 = np.sum(beta3_grid2[0, 0, 0, :] * marg_sigma2_2) * dk2
print 'Segundo modelo: Gaussiana doble'
print 'Amplitud 1               :', E_A2_1
print 'sigma 1                  :', E_sigma2_1
print ''
print 'Amplitud 2               :', E_A2_2
print 'sigma 2                  :', E_sigma2_2
print ''
print "Factor bayesiano:", P_E1/P_E2

'''
-----------------------------------------------------------------------------
Guardando informacion para procesar en otro script:
'''
#len: 201, 201, 51, 51, 51, 51
marginales = np.concatenate((marg_A1, marg_sigma1, marg_A2_1, marg_sigma2_1,
                             marg_A2_2, marg_sigma2_2), axis=0)
esperanzas = [E_A1, E_sigma1, E_A2_1, E_sigma2_1, E_A2_2, E_sigma2_2]
dmarginales = np.concatenate((beta0_grid1[:, 0], beta1_grid1[0, :],
                              beta0_grid2[:, 0, 0, 0], beta1_grid2[0, :, 0, 0],
                              beta2_grid2[0, 0, :, 0],
                              beta3_grid2[0, 0, 0, :]), axis=0)

np.savetxt('marginales.txt', marginales)
np.savetxt('esperanzas.txt', esperanzas)
np.savetxt('dmarginales.txt', dmarginales)
