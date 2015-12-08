#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Este script presenta los resultados obtenidos a partir de bayes.py
'''

import numpy as np
import matplotlib.pyplot as plt

wavelen = np.loadtxt("espectro.dat", usecols=(0,))
frec = np.loadtxt("espectro.dat", usecols=(1,))
E = np.load('scBayesianyE.npy')
scala = E[0]
bayes = E[1]
A1 = np.load('margAmp1.npy')
domA1 = np.load('dmargAmp1.npy')
S1 = np.load('margstd1.npy')
domS1 = np.load('dmargstd1.npy')
A21 = np.load('margAmp21.npy')
domA21 = np.load('dmargAmp21.npy')
S21 = np.load('margstd21.npy')
domS21 = np.load('dmargstd21.npy')
A22 = np.load('margAmp22.npy')
domA22 = np.load('dmargAmp22.npy')
S22 = np.load('margstd22.npy')
domS22 = np.load('dmargstd22.npy')
doms = [domA1, domS1, domA21, domS21, domA22, domS22]
margins = [A1, S1, A21, S21, A22, S22]
emargins = [E[2], E[3], E[4], E[5], E[6], E[7]]
cred_inter = []

# Cálculo de intervalos de credibilidad
for i in range(len(margins)):
    dx = doms[i][1] - doms[i][0]
    for n in range(len(doms[i])):
        if doms[i][n] >= emargins[i]:
            left = n - 1
            right = n + 1
            break
    s = dx * margins[i][n]
    while s < 0.68:
        if margins[i][right] > margins[i][left]:
            s += dx * margins[i][right]
            right += 1
        else:
            s += dx * margins[i][left]
            left -= 1
        if margins[i][right] > margins[i][left] and s < 0.68:
            xl = doms[i][left]
            while margins[i][right] > margins[i][left]:
                s += dx * margins[i][right]
                right += 1
            xr = (doms[i][right-1] + (doms[i][right]-doms[i][right-1])*
                  (margins[i][left]-margins[i][right-1]) /
                  (margins[i][right]-margins[i][right-1]))
        elif s < 0.68:
            xr = doms[i][right]
            while margins[i][left] >= margins[i][right]:
                s += dx * margins[i][left]
                left -= 1
            xl = (doms[i][left] + (doms[i][left+1]-doms[i][left])*
                  (margins[i][right]-margins[i][left]) /
                  (margins[i][left+1]-margins[i][left]))
    cred_inter.append((xl, xr))

cred_inter[0] = (cred_inter[0][0]/scala, cred_inter[0][1]/scala)
cred_inter[2] = (cred_inter[2][0]/scala, cred_inter[2][1]/scala)
cred_inter[4] = (cred_inter[4][0]/scala, cred_inter[4][1]/scala)

print "Factor bayesiano P(D|M1) / P(D|M2) =", bayes
print "Amplitud modelo 1:", emargins[0]/scala, "[erg s^-1 Hz^-1 cm^-2]"
print "Inter. de cred.:", cred_inter[0], "[erg s^-1 Hz^-1 cm^-2]"
print "Ancho modelo 1:", emargins[1], "[A]"
print "Inter. de cred.:", cred_inter[1], "[A]"
print ""
print "Amplitud 1 modelo 2:", emargins[2]/scala, "[erg s^-1 Hz^-1 cm^-2]"
print "Inter. de cred.:", cred_inter[2], "[erg s^-1 Hz^-1 cm^-2]"
print "Ancho 1 modelo 2:", emargins[3], "[A]"
print "Inter. de cred.:", cred_inter[3], "[A]"
print "Amplitud 2 modelo 2:", emargins[4]/scala, "[erg s^-1 Hz^-1 cm^-2]"
print "Inter. de cred.:", cred_inter[4], "[erg s^-1 Hz^-1 cm^-2]"
print "Ancho 2 modelo 2:", emargins[5], "[A]"
print "Inter. de cred.:", cred_inter[5], "[A]"

# Plots
plt.clf()
plt.figure(1)
plt.plot(domA1/scala, A1*scala)
plt.title('Densidad de probabilidad, amplitud, modelo 1')
plt.xlabel('Amplitud [erg $s^{-1}$ $Hz^{-1}$ $cm^{-2}$]')
plt.axvline(x=emargins[0]/scala, color='g')
plt.axvline(x=cred_inter[0][0], color='r')
plt.axvline(x=cred_inter[0][1], color='r')
plt.savefig('A1.eps')

plt.figure(2)
plt.plot(domS1, S1)
plt.title('Densidad de probabilidad, ancho de curva, modelo 1')
plt.xlabel('$\sigma$ [$\AA$]')
plt.axvline(x=emargins[1], color='g')
plt.axvline(x=cred_inter[1][0], color='r')
plt.axvline(x=cred_inter[1][1], color='r')
plt.savefig('S1.eps')

plt.figure(3)
plt.plot(domA21/scala, A21*scala)
plt.title('Densidad de probabilidad, amplitud 1, modelo 2')
plt.xlabel('Amplitud [erg $s^{-1}$ $Hz^{-1}$ $cm^{-2}$]')
plt.axvline(x=emargins[2]/scala, color='g')
plt.axvline(x=cred_inter[2][0], color='r')
plt.axvline(x=cred_inter[2][1], color='r')
plt.savefig('A21.eps')

plt.figure(4)
plt.plot(domS21, S21)
plt.title('Densidad de probabilidad, ancho de curva 1, modelo 2')
plt.xlabel('$\sigma$ [$\AA$]')
plt.axvline(x=emargins[3], color='g')
plt.axvline(x=cred_inter[3][0], color='r')
plt.axvline(x=cred_inter[3][1], color='r')
plt.savefig('S21.eps')

plt.figure(5)
plt.plot(domA22/scala, A22*scala)
plt.title('Densidad de probabilidad, amplitud 2, modelo 2')
plt.xlabel('Amplitud [erg $s^{-1}$ $Hz^{-1}$ $cm^{-2}$]')
plt.axvline(x=emargins[4]/scala, color='g')
plt.axvline(x=cred_inter[4][0], color='r')
plt.axvline(x=cred_inter[4][1], color='r')
plt.savefig('A22.eps')

plt.figure(6)
plt.plot(domS22, S22)
plt.title('Densidad de probabilidad, ancho de curva 2, modelo 2')
plt.xlabel('$\sigma$ [$\AA$]')
plt.axvline(x=emargins[5], color='g')
plt.axvline(x=cred_inter[5][0], color='r')
plt.axvline(x=cred_inter[5][1], color='r')
plt.savefig('S22.eps')
