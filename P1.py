import numpy as np
import matplotlib.pyplot as plt
import modelo1 as mod


# Main driver
if __name__ == '__main__':

    # Importar datos
    data = np.loadtxt("espectro.dat", dtype='float', skiprows=3)
    wavelength = data[:,0]
    flux = data[:,1]

    scale_fact = 2.5e18
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
    std_b = (wavelength[1] - wavelength[0])

    ''' Crear modelo '''
    modelo = mod.make_modelo1(var_data, mean_A, std_A, mean_b, std_b, C_0, lambda_0)

    modelo.normalizar(wavelength, flux)  # Permite usar prob a posteriori norm.

    ''' Obtener valor optimo de A '''
    x_A = modelo.full_A
    y_A = modelo.full_A * 0.0
    for i in range(len(x_A)):
        y_A[i] = modelo.prob_posteriori_A(wavelength, flux, x_A[i])

    optimal_A = sum(x_A * y_A) / sum(y_A)
    print('El mejor valor de A es: ' + str(optimal_A / scale_fact))

    # Buscar indice mas cercano a valor optimo
    target = optimal_A
    vals = abs(x_A - target)
    index_mean = vals.argmin()

    # Buscar intervalo confianza de A
    delta_x = x_A[1] - x_A[0]
    suma = y_A[index_mean] * delta_x
    delta_index = 1
    K = sum(y_A * delta_x)

    while suma < 0.68:
        # Se avanza un paso a derecha e izquierda en torno al optimo
        suma += y_A[index_mean + delta_index] / K * delta_x
        suma += y_A[index_mean - delta_index] / K * delta_x
        delta_index += 1

    min_A = x_A[index_mean - delta_index] / scale_fact
    max_A = x_A[index_mean + delta_index] / scale_fact
    print('El intervalo de confianza de A es: '
    + str(min_A) + ' -- ' + str(max_A))

    # Graficar distribucion de probabilidad de A
    plt.figure()
    plt.plot(x_A, y_A, 'r')

    ''' Obtener valor optimo de b'''
    x_b = modelo.full_b
    y_b = modelo.full_b * 0.0
    for i in range(len(x_b)):
        y_b[i] = modelo.prob_posteriori_b(wavelength, flux, x_b[i])

    optimal_b = sum(x_b * y_b) / sum(y_b)
    print('El mejor valor de b es: ' + str(optimal_b))

    # Buscar indice mas cercano a valor optimo
    target = optimal_b
    vals = abs(x_b - target)
    index_mean = vals.argmin()

    # Buscar intervalo de confianza de b
    delta_x = x_b[1] - x_b[0]
    suma = y_b[index_mean] * delta_x
    delta_index = 1
    K = sum(y_b * delta_x)

    while suma < 0.68:
        # Se avanza un paso a derecha e izquierda en torno al optimo
        suma += y_b[index_mean + delta_index] / K * delta_x
        suma += y_b[index_mean - delta_index] / K * delta_x
        delta_index += 1

    min_b = x_b[index_mean - delta_index]
    max_b = x_b[index_mean + delta_index]
    print('El intervalo de confianza de b es: '
    + str(min_b) + ' -- ' + str(max_b))

    # Graficar probabilidad de b
    plt.figure()
    plt.plot(x_b, y_b, 'b')


    ''' Graficar modelo vs datos '''
    x_modelo = np.linspace(wavelength[0], wavelength[-1], 200)
    plt.figure()
    plt.plot(wavelength, flux / scale_fact, 'o')
    plt.plot(x_modelo, modelo.model(x_modelo, optimal_A, optimal_b) / scale_fact)
    plt.show()
