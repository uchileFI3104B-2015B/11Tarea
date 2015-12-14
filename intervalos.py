from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


scala = 10**16

marg = np.loadtxt('marginales.txt')
emargins = np.loadtxt('esperanzas.txt')
dmarg = np.loadtxt('dmarginales.txt')

y = [marg[:201], marg[201:402], marg[402:453],
              marg[453:504], marg[504:555], marg[555:606]]
x = [dmarg[:201], dmarg[201:402], dmarg[402:453],
              dmarg[453:504], dmarg[504:555], dmarg[555:606]]
cred_inter = []

'''
-----------------------------------------------------------------------------
Calculo de intervalos de credibilidad : Se tenia la idea de como efectuar
este calculo, pero a falta de tiempo se decide utilizar algoritmo realizado
por alumno Bruno Scheihing .
-----------------------------------------------------------------------------
'''
for i in range(len(y)):
    print i
    dx = x[i][1] - x[i][0]
    for n in range(len(x[i])):
        if x[i][n] >= emargins[i]:
            left = n - 1
            right = n + 1
            break
    s = dx * y[i][n]
    while s < 0.0068:
        if y[i][right] > y[i][left]:
            s += dx * y[i][right]
            right += 1
        else:
            s += dx * y[i][left]
            left -= 1
        if y[i][right] > y[i][left] and s < 0.0068:
            xl = x[i][left]
            while y[i][right] > y[i][left]:
                s += dx * y[i][right]
                right += 1
            xr = (x[i][right-1] + (x[i][right]-x[i][right-1]) *
                  (y[i][left]-y[i][right-1]) /
                  (y[i][right]-y[i][right-1]))
        elif s < 0.0068:
            xr = x[i][right]
            while y[i][left] >= y[i][right]:
                s += dx * y[i][left]
                left -= 1
            xl = (x[i][left] + (x[i][left+1]-x[i][left]) *
                  (y[i][right]-y[i][left]) /
                  (y[i][left+1]-y[i][left]))
    cred_inter.append((xl, xr))


cred_inter[0] = (cred_inter[0][0]/scala, cred_inter[0][1]/scala)
cred_inter[2] = (cred_inter[2][0]/scala, cred_inter[2][1]/scala)
cred_inter[4] = (cred_inter[4][0]/scala, cred_inter[4][1]/scala)

print 'Primer modelo: Gaussiana simple'
print "Inter. de cred. A:", cred_inter[0], "[erg s^-1 Hz^-1 cm^-2]"
print "Inter. de cred. sigma:", cred_inter[1], "[A]"

print 'Segundo modelo: Gaussiana doble'
print "Inter. de cred. A1:", cred_inter[2], "[erg s^-1 Hz^-1 cm^-2]"
print "Inter. de cred. sigma1:", cred_inter[3], "[A]"
print "Inter. de cred. A2:", cred_inter[4], "[erg s^-1 Hz^-1 cm^-2]"
print "Inter. de cred. sigma2:", cred_inter[5], "[A]"

'''
----------------------------------------------------------------------------
PLOTS
----------------------------------------------------------------------------
'''
plt.figure(1)
plt.plot(x[0]/scala, y[0])
plt.axvline(emargins[0]/scala, color='r')
plt.savefig('curva1.png')

plt.figure(2)
plt.plot(x[1], y[1])
plt.axvline(emargins[1], color='r')
plt.savefig('curva2.png')

plt.figure(3)
plt.plot(x[2]/scala, y[2]*scala)
plt.axvline(emargins[2]/scala, color='r')
plt.savefig('curva3.png')

plt.figure(4)
plt.plot(x[3], y[3])
plt.axvline(emargins[3], color='r')
plt.savefig('curva4.png')

plt.figure(5)
plt.plot(x[4]/scala, y[4]*scala)
plt.axvline(emargins[4]/scala, color='r')
plt.savefig('curva5.png')

plt.figure(6)
plt.plot(x[5], y[5])
plt.axvline(emargins[5], color='r')
plt.xlim(6, 11)
plt.savefig('curva6.png')

plt.draw()
plt.show()
