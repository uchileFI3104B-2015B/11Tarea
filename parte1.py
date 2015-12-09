from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pymc3 as pm
from theano import function, shared
import theano.tensor as tt


#funciones estructurales

def leer_archivo(nombre):
    '''
    lee el archivo
    nombre debe ser un str
    '''
    datos = np.loadtxt(nombre)
    x = datos[:, 0]
    y = datos[:, 1]
    return x, y


def gauss(mu, sigma, x):
    y = tt.exp(tt.as_tensor_variable(- (x - mu) ** 2 / (2 * sigma ** 2)))
    f = function([], y)
    return f()


def modelo_doble(A1, sigma1, A2, sigma2, x_sample):
    a = 10 ** (- 16)
    mu = 6563
    y = tt.as_tensor_variable(a - A1 * gauss(mu, sigma1, x_sample) - A2 * gauss(mu, sigma2, x_sample))
    f = function([], y)
    return f()


'''
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
'''


# main
x_sample, y_sample = leer_archivo('espectro.dat')
# inicializacion
A1 = 1e-17
A2 = 1e-17
sigma1 = 7.
sigma2 = 7.
a = 10 ** (- 16)
mu = 6563
p = A1, sigma1, A2, sigma2
with pm.Model() as basic_model:
    # priors
    beta0 = pm.Normal('beta0', mu=A1, sd=1)
    beta1 = pm.Normal('beta1', mu=sigma1, sd=1)
    beta2 = pm.Normal('beta2', mu=A2, sd=1)
    beta3 = pm.Normal('beta3', mu=sigma2, sd=1)
    # valor esperado
    #y_out = modelo_2(p1, x_sample)
    #y_out = tt.as_tensor_variable(10 ** (- 16)) - tt.as_tensor_variable(beta0) * pm.Normal('a', mu=6563, sd=beta1) - tt.as_tensor_variable(beta2) * pm.Normal('b', mu=6563, sd=beta2)
    y_out = tt.as_tensor_variable(10 ** (- 16) - beta0 * tt.exp(tt.as_tensor_variable(- (x_sample - 6563) ** 2 / (2 * beta1 ** 2))) - beta2 * tt.exp(tt.as_tensor_variable(- (x_sample - 6563) ** 2 / (2 * beta3 ** 2))))
    #y_out = modelo_doble(beta0, beta1, beta2, beta3, x_sample)
    #y_out = a - beta0 * gauss(mu, beta1, x_sample) - beta2 * gauss(mu, beta3, x_sample)
    # likelihood
    Y_obs = pm.Normal('Y_obs', mu=y_out, sd=1.5, observed=y_sample)
    map_estimate = pm.find_MAP(model=basic_model)
    print map_estimate
'''
with basic_model:
    trace = pm.sample(5000, start=map_estimate)
plt.pcolormesh(beta0_grid, beta1_grid, likelihood_grid * prior_grid, cmap='viridis')
plt.xlim(-5, 11)
plt.ylim(1,  6)
plt.plot(trace.beta0, trace.beta1, marker='None', ls='-', lw=0.3, color='w')
'''
