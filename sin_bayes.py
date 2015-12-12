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


def gauss(x, A, sigma1):
    '''
    Retorna el valor de la suma de dos gaussianas evaluadas en x.
    '''
    g = (A * scipy.stats.norm(loc=6563, scale=sigma1).pdf(x))
    return 1e-16 - g


def chi(func, x, y, parametros):
    chi2 = 0
    for i in range(len(x)):
        chi2 += (y[i] - func(x[i], *parametros))**2
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
#                                                                           #
#############################################################################

# Leer datos
x, y = llamar_archivo('espectro.dat')

# Puntos escogidos como inicio de la linea de absorcion
p.plot(x[50], y[50], '*')
p.plot(x[76], y[76], '*')
p.plot(np.linspace(np.min(x), np.max(x), 2), np.ones(2)*1e-16, 'r', lw=1.5)
p.plot(x, y, color='turquoise', drawstyle='steps-post', lw=2., alpha=0.8)
p.axis([6503, 6623, 9e-17, 1.01e-16])
p.xlabel('Angstrom')
p.ylabel('erg / s / Hz / cm^2')
p.grid(True)
p.show()

# Semillas para el ajuste
seeds = [1e-17, 7., 1e-17, 7.]

# Resultados
resultados_gauss(x, y, seeds)
print 'Desviacion STD fuera de la linea de absorcion = ', desviacion(y, 50, 76)
