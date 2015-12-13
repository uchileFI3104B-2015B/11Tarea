from __future__ import division
import scipy.stats
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def model_1(x, A, sigma):
    return 10**-16 - A * scipy.stats.cauchy(loc=6563, scale=sigma).pdf(x)


def model_2(x, A1, sigma1, A2, sigma2):
    gaussiana1 = (A1 * scipy.stats.norm(loc=6563, scale=sigma1).pdf(x))
    gaussiana2 = (A2 * scipy.stats.norm(loc=6563, scale=sigma2).pdf(x))
    return 10**-16 - gaussiana1 - gaussiana2

def chi2(model, params, data):
    x_datos = data[0]
    y_datos = data[1]
    y_model = model(x_datos, *params)
    chi2 = (y_datos - y_model)**2
    return np.sum(chi2)

wavelength = np.loadtxt("espectro.dat", usecols=[0])
fnu = np.loadtxt("espectro.dat", usecols=[1])
x = np.linspace(min(wavelength), max(wavelength), 100)
params_optimos_1 = [0.75 , 3.6534]  # obtenidos de otro script
#params_optimos_2 = [ ] 


# plot 1
fig = plt.figure(1)
fig.clf()
plt.plot(wavelength, fnu, color='g', drawstyle='steps-post',
         label='Datos')
plt.plot(x, model_1(x, *params_optimos), label='Modelo 1 (Gaussiana simple)')
plt.legend()
plt.draw()
plt.show()

# plot 2
fig2 = plt.figure(2)
