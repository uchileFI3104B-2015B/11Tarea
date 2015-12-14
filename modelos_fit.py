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


wavelength = np.loadtxt("espectro.dat", usecols=[0])
fnu = np.loadtxt("espectro.dat", usecols=[1])
x = np.linspace(min(wavelength), max(wavelength), 200)


# Los siguientes datos fueron obtenidos con tarea.py
params_optimos_1 = [0.7619798 , 3.7052664]
params_optimos_2 = [0.413796, 2.4562, 0.482985, 8.502471]
pars1 = [0.413796, 2.4562]
pars2 = [0.482985, 8.502471]


# plot 1
fig = plt.figure(1)
fig.clf()
plt.plot(wavelength, fnu, 'o', color='g', label='Datos', alpha=0.7)
plt.plot(x, model_1(x, *params_optimos_1), label='Modelo 1', color='r')
plt.legend(loc=4)
plt.draw()
plt.show()
plt.savefig('figura1.png')

# plot 2
fig2 = plt.figure(2)
fig2.clf()
plt.plot(wavelength, fnu, 'o', color='g', label='Datos', alpha=0.7)
plt.plot(x, model_2(x, *params_optimos_2), label='Modelo 2', color='b')
plt.plot(x, model_1(x, *pars1), 'b--')
plt.plot(x, model_1(x, *pars2), 'b--')
plt.legend(loc=4)
plt.draw()
plt.show()
plt.savefig('figura2.png')

# plot 3
fig3 = plt.figure(3)
fig3.clf()
plt.plot(wavelength, fnu, 'o', color='g', label='Datos', alpha=0.7)
plt.plot(x, model_1(x, *params_optimos_1), label='Modelo 1', color='r')
plt.plot(x, model_2(x, *params_optimos_2), label='Modelo 2', color='b')
plt.legend(loc=4)
plt.draw()
plt.show()
plt.savefig('figura3.png')
