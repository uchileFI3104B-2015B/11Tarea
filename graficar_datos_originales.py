import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("espectro.dat", dtype='float', skiprows=3)
wavelength = data[:,0]
flux = data[:,1]

''' Graficar datos originales '''
plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
plt.plot(wavelength, flux, 'r')
plt.xlim([wavelength[0], wavelength[-1]])
plt.xlabel('Wavelength$[\AA]$', size=16)
plt.ylabel('$F_\upsilon[erg\ s^{-1} Hz^{-1}cm^{-2}]$', size=16)
plt.title('Observacion Espectroscopica', size=16, y=1.04)
plt.grid()
plt.show()
