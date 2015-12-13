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
que muesta continuo y una linea de absorcion, el continuo esta dado y la linea
de absorcion se modela con una doble gaussiana
'''


def datos():
    '''
    Descarga los datos de un archivo y los retorna en columnas
    '''
    lambda1 = np.loadtxt('espectro.dat', usecols=(-2,))
    flujo = np.loadtxt('espectro.dat', usecols=(-1,))
    return lambda1, flujo

def gaussiana2(x, A, A2, sigma, sigma2):
    '''
    modela linea de absorsion a traves de doble gaussiana
    '''
    return 1e-16 - (A * scipy.stats.cauchy(loc=6563, scale=sigma).pdf(x) +
            A2 * scipy.stats.cauchy(loc=6563, scale=sigma2).pdf(x))


a2 = 0.02e-16, 0.04e-16, 6, 3
