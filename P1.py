import numpy as np
import matplotlib.pyplot as plt
import modelo1 as mod

# Main driver
if __name__ == '__main__':

    # Importar datos
    data = np.loadtxt("espectro.dat", dtype='float', skiprows=3)
    wavelength = data[:,0]
    flux = data[:,1]

    scale_fact = 2.45e18
    flux = flux * scale_fact
    C_0 = 1e-16 * scale_fact  # Nivel del continuo
    lambda_0 = 6563

    ''' Determinar error de medicion '''
    suma = 0.0
    n = 0
    for i in range(len(flux)):

        mean_continuo = C_0

        if wavelength[i] < 6525 or wavelength[i] > 6600:
            n += 1
            suma += (flux[i] - mean_continuo) ** 2.0

    var_data = suma / n  # Varianza medicion

    ''' Determinar parametros distribuciones de A y b '''
    ''' Parametro A '''
    mean_A = abs(min(flux) - C_0)
    std_A = np.sqrt(var_data)  # Estimador sesgado de desv. estandar

    ''' Parametro b '''
    # Se busca valor que cumple condicion
    target = C_0 - mean_A / 2.0
    values = abs(flux - target)
    index_min = values.argmin()

    mean_b = abs(lambda_0 - wavelength[index_min])
    std_b = (wavelength[1] - wavelength[0]) / 2

    ''' Crear modelo '''
    modelo = mod.make_modelo1(var_data, mean_A, std_A, mean_b, std_b, C_0, lambda_0)

    modelo.normalizar(wavelength, flux)  # Permite usar prob a posteriori norm.

    p = modelo.prob_a_posteriori(wavelength, flux, mean_A, mean_b)
    print(p)

    ''' Graficar modelo1 vs long de onda '''
    x = np.linspace(min(wavelength), max(wavelength), 1000)
    y = modelo.model(x, mean_A, mean_b*1.2)

    plt.plot(wavelength, flux, 'r')
    plt.plot(x, y, 'b')
    plt.show()
