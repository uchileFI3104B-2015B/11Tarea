# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import (leastsq, curve_fit)
from scipy import optimize as opt
from scipy.stats import kstest
import os
from scipy.stats import kstwobign

'''
Este script modela 2 parametros a la vez a partir de los datos de
espectro.dat, al plotear se observa un segmento del espectro de una fuente
que muesta continuo (con una leve pendiente) y una linea de absorcion, se
modela el primero con una linea recta y el segundo con una
gaussiana o doble gaussiana
'''


def datos():
    '''
    Descarga los datos de un archivo y los retorna en columnas
    '''
    lambda1 = np.loadtxt('espectro.dat', usecols=(-2,))
    flujo = np.loadtxt('espectro.dat', usecols=(-1,))
    return lambda1, flujo


def linea():
    '''
    modela continuo a traves de una linea recta
    '''
    return 1e-16


def gaussiana(x, A, sigma):
    '''
    modela linea de absorsion a traves de una gaussiana
    '''
    return A * scipy.stats.norm(loc=6563, scale=sigma).pdf(x)


def gaussiana2(x, A, A2, sigma, sigma2):
    '''
    modela linea de absorsion a traves de doble gaussiana
    '''
    return A * scipy.stats.cauchy(loc=6563, scale=sigma).pdf(x) +
            A2 * scipy.stats.cauchy(loc=6563, scale=sigma2).pdf(x)

def modelo1gaus(x, A, sigma):
    '''
    Modelo con linea de absorcion con una gaussiana
    '''
    return linea() - gaussiana(x, A, sigma)

def modelo2gaus(x, A, A2, sigma, sigma2):
    '''
    Modelo con linea de absorcion con doble gaussiana
    '''
    return linea() - gaussiana2(x, A, A2, sigma, sigma2)

# Main

a1 = 0.08e-16, 3
a2 = 0.02e-16, 0.04e-16, 6, 3
