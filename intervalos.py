from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


escala = 10**16

marg = np.loadtxt('marginales.txt')
emargins = np.loadtxt('esperanzas.txt')
dmarg = np.loadtxt('dmarginales.txt')

y = [marg[:201], marg[201:402], marg[402:453],
              marg[453:504], marg[504:555], marg[555:606]]
x = [dmarg[:201], dmarg[201:402], dmarg[402:453],
              dmarg[453:504], dmarg[504:555], dmarg[555:606]]
intervalo = []

'''
-----------------------------------------------------------------------------
Calculo de intervalos de credibilidad: Dado que primero se realizo el ploteo,
nos aprovechamos de que las curvas son bastante simetricas para luego
decir que podemos evaluar tanto en left como el right en la fila 42. Tambien
nos aprovechamos de lo anterior para ir corriendo el left y el right al mismo
tiempo en una unidad. No es lo optimo, pero es una buena aproximacion.
-----------------------------------------------------------------------------
'''
# dado las diferencias de escala se calcularon a "mano" los quivalentes
# a 68%.
equivalente = [0.002296, 0.098, 0.0467, 0.10014, 0.0643, 0.214]
for i in range(len(y)):
    dx = x[i][1] - x[i][0]
    for n in range(len(x[i])):
        if x[i][n] >= emargins[i]:
            left = n - 1
            right = n + 1
            print n
            break
    s = dx * y[i][n]
    while s < equivalente[i]:
        left -=1
        right +=1
        s += dx * y[i][right]
        print s
    xl = x[i][left]
    xr = x[i][right]
    intervalo.append((xl, xr))


intervalo[0] = (intervalo[0][0]/escala, intervalo[0][1]/escala)
intervalo[2] = (intervalo[2][0]/escala, intervalo[2][1]/escala)
intervalo[4] = (intervalo[4][0]/escala, intervalo[4][1]/escala)

print 'Intervalos primer modelo: Gaussiana simple'
print "A:", intervalo[0], "[erg s^-1 Hz^-1 cm^-2]"
print "sigma:", intervalo[1], "[A]"

print 'Intervalos segundo modelo: Gaussiana doble'
print "A1:", intervalo[2], "[erg s^-1 Hz^-1 cm^-2]"
print "sigma1:", intervalo[3], "[A]"
print "A2:", intervalo[4], "[erg s^-1 Hz^-1 cm^-2]"
print "sigma2:", intervalo[5], "[A]"

'''
----------------------------------------------------------------------------
PLOTS
----------------------------------------------------------------------------
'''
plt.clf()
plt.figure(1)
plt.plot(x[0]/escala, y[0]*escala, color='b')
plt.axvline(emargins[0]/escala, color='g', label='Esperanza')
plt.axvline(intervalo[0][0], color='r', label='Intervalo credibilidad')
plt.axvline(intervalo[0][1], color='r')
plt.title('Densidad de probabilidad $A$, modelo 1')
plt.xlabel('Amplitud [$erg$ $s^{-1}$ $Hz^{-1}$ $cm^{-2}$]')
plt.legend()
plt.savefig('curva1.png')

plt.figure(2)
plt.plot(x[1], y[1], color='b')
plt.axvline(emargins[1], color='g', label='Esperanza')
plt.axvline(intervalo[1][0], color='r', label='Intervalo credibilidad')
plt.axvline(intervalo[1][1], color='r')
plt.title('Densidad de probabilidad $\sigma$, modelo 2')
plt.legend(loc=2)
plt.xlabel('$\sigma$ [$\AA$]')
plt.savefig('curva2.png')

plt.figure(3)
plt.plot(x[2]/escala, y[2]*escala, color='b')
plt.axvline(emargins[2]/escala, color='g', label='Esperanza')
plt.axvline(intervalo[2][0], color='r', label='Intervalo credibilidad')
plt.axvline(intervalo[2][1], color='r')
plt.title('Densidad de probabilidad $A_1$, modelo 2')
plt.xlabel('Amplitud [$erg$ $s^{-1}$ $Hz^{-1}$ $cm^{-2}$]')
plt.legend()
plt.savefig('curva3.png')

plt.figure(4)
plt.plot(x[3], y[3], color='b')
plt.axvline(emargins[3], color='g', label='Esperanza')
plt.axvline(intervalo[3][0], color='r', label='Intervalo credibilidad')
plt.axvline(intervalo[3][1], color='r')
plt.title('Densidad de probabilidad $\sigma_1$, modelo 2')
plt.legend()
plt.xlabel('$\sigma$ [$\AA$]')
plt.savefig('curva4.png')

plt.figure(5)
plt.plot(x[4]/escala, y[4]*escala, color='b')
plt.axvline(emargins[4]/escala, color='g', label='Esperanza')
plt.axvline(intervalo[4][0], color='r', label='Intervalo credibilidad')
plt.axvline(intervalo[4][1], color='r')
plt.legend(loc=2)
plt.title('Densidad de probabilidad $A_1$, modelo 2')
plt.xlabel('Amplitud [$erg$ $s^{-1}$ $Hz^{-1}$ $cm^{-2}$]')
plt.savefig('curva5.png')

plt.figure(6)
plt.plot(x[5], y[5], color='b')
plt.axvline(emargins[5], color='g', label='Esperanza')
plt.axvline(intervalo[5][0], color='r', label='Intervalo credibilidad')
plt.axvline(intervalo[5][1], color='r')
plt.xlabel('$\sigma$ [$\AA$]')
plt.legend()
plt.title('Densidad de probabilidad $\sigma_2$, modelo 2')
plt.savefig('curva6.png')

plt.draw()
plt.show()
