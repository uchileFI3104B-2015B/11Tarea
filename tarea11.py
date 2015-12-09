#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit

'''
En este código se busca modelar el ensanchamiento de una linea de absorcion
en un espectro de radiación.
Para ello se utilizan dos modelos: uno gaussiano simple y otro con dos
funciones gaussianas
'''

def modelo_1(x, A, sigma):
    '''
    Retorna la función correspondiente al primer modelo, es decir, una
    gaussiana trasladada en 1e-16
    '''
    return 1e-16 - A * scipy.stats.cauchy(loc=6563, scale=sigma).pdf(x)

def modelo_2(x, Amplitud1, sigma1, Amplitud2, sigma2):
    '''
    Retorna la función correspondiente al primer modelo, es decir, la suma
    de dos gaussianas trasladada en 1e-16
    '''
    gauss1 = (Amplitud1 * scipy.stats.norm(loc=6563, scale=sigma1).pdf(x))
    gauss2 = (Amplitud2 * scipy.stats.norm(loc=6563, scale=sigma2).pdf(x))
    return 1e-16 - gauss1 - gauss2

def chi2(data, parametros_modelo, funcion_modelo):
    x_datos = data[0]
    y_datos = data[1]
    y_modelo = funcion_modelo(x_datos, *parametros_modelo)
    chi2 = (y_datos - y_modelo)**2
    return np.sum(chi2)

# Main
# Datos
wavelength = np.loadtxt("espectro.dat", usecols=[0])
Fnu = np.loadtxt("espectro.dat", usecols=[1])
x = np.linspace(min(wavelength), max(wavelength), 100)  #para plotear

# Modelo 1:
Amplitud_mod1 = Fnu.max() - Fnu.min()
sigma_mod1 = 6 # Del gráfico
adivinanza_mod1 = [Amplitud_mod1, sigma_mod1]
param_optimo1, param_covar1 = curve_fit(modelo_1, wavelength, Fnu,
                                        adivinanza_mod1)
Amplitud_mod1, sigma_mod1 = param_optimo1
chi2_1 = chi2([wavelength, Fnu], param_optimo1, modelo_1)

print 'Primer modelo: Gaussiana simple'
print 'Amplitud                :', Amplitud_mod1
print 'sigma                   :', sigma_mod1
print 'Chi2                    :', chi2_1
print ''

# Modelo 2:
Amplitud1_mod2 = 0.02e-16
sigma1_mod2 = 6.
Amplitud2_mod2 = 0.07e-16
sigma2_mod2 = 1.5
adivinanza_mod2 = [Amplitud1_mod2, sigma1_mod2, Amplitud2_mod2, sigma2_mod2]
param_optimo2, param_covar2 = curve_fit(modelo_2, wavelength, Fnu,
                                        adivinanza_mod2)
Amplitud1_mod2, sigma1_mod2, Amplitud2_mod2, sigma2_mod2 = param_optimo2
chi2_2 = chi2([wavelength, Fnu], param_optimo2, modelo_2)

print 'Primer modelo: Gaussiana simple'
print 'Amplitud 1              :', Amplitud1_mod2
print 'sigma 1                 :', sigma1_mod2
print 'Amplitud 2              :', Amplitud2_mod2
print 'sigma 2                 :', sigma2_mod2
print 'Chi2                    :', chi2_2
print ''


# Plots
plt.figure(1, figsize=(10,7))
plt.style.use('bmh')
plt.plot(wavelength, Fnu, color='brown', drawstyle='steps-post',
         label='Datos')
plt.plot(x, modelo_1(x, *param_optimo1), label='Modelo 1 (Gaussiana simple)',
                     linewidth=2.0)
plt.plot(x, modelo_2(x, *param_optimo2), '--', color='fuchsia',  linewidth=2.0,
         label='Modelo 2 (Gaussiana doble)')
plt.plot(x, modelo_1(x, Amplitud1_mod2, sigma1_mod2), '--', color='g',
         label='Gaussiana 1 (modelo 2)', alpha=0.8)
plt.plot(x, modelo_1(x, Amplitud2_mod2, sigma2_mod2), '-.', color='g',
         label='Gaussiana (modelo 2)', alpha=0.8)
plt.xlabel('Wavelength [$\AA$]', fontsize=16)
plt.ylabel('$F_v$[erg s$^{-1}$Hz$^{-1}$cm$^{-2}$]', fontsize=16)
plt.xlim(6520, 6600)
plt.ylim(0.9e-16, 1.02e-16)
plt.legend(loc='lower left')
plt.grid(False)
#plt.savefig('Fits_espectro.eps')

plt.show()
