from __future__ import division
import scipy.stats
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2*sigma**2))/np.sqrt(2*np.pi*sigma**2)


def model_1(x, A, sigma):
    return 10**-16 - A * (10**-16) * gauss(x, 6563, sigma)


def model_2(x, A1, sigma1, A2, sigma2):
    gaussiana1 = 10**-16 - A1 * (10**-16) * gauss(x, 6563, sigma1)
    gaussiana2 = 10**-16 - A2 * (10**-16) * gauss(x, 6563, sigma2)
    return - 10**-16 + gaussiana1 + gaussiana2

def chi2(model, params, data):
    x_datos = data[0]
    y_datos = data[1]
    y_model = model(x_datos, *params)
    chi2 = (y_datos - y_model)**2
    return np.sum(chi2)

wavelength = np.loadtxt("espectro.dat", usecols=[0])
fnu = np.loadtxt("espectro.dat", usecols=[1])
x = np.linspace(min(wavelength), max(wavelength), 200)
params_optimos_1 = [0.761611 , 3.703762]  # obtenidos de otro script
params_optimos_2 = [0.413656, 2.455765, 0.483097, 8.500034]
pars1 = [0.413656, 2.455765]
pars2 = [0.483097, 8.500034]


# plot 1
fig = plt.figure(1)
fig.clf()
plt.plot(wavelength, fnu, 'o', color='g', label='Datos', alpha=0.7)
plt.plot(x, model_1(x, *params_optimos_1), label='Modelo 1', linewidth=1.5)
plt.legend(loc=4)
plt.draw()
plt.show()
plt.savefig('figura1.png')

# plot 2
fig2 = plt.figure(2)
fig2.clf()
plt.plot(wavelength, fnu, 'o', color='g', label='Datos', alpha=0.7)
plt.plot(x, model_2(x, *params_optimos_2), label='Modelo 2',
         color='b', linewidth=1.5)
plt.plot(x, model_1(x, *pars1), 'b--')
plt.plot(x, model_1(x, *pars2), 'b--')
plt.legend(loc=4)
plt.draw()
plt.show()
plt.savefig('figura2.png')
