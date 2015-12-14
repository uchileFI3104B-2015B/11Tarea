import numpy as np
from scipy import integrate


class modelo1(object):

    ''' Clase que permite crear cada una de las funciones asociadas
    al modelo 1 y modificar sus parametros '''

    var_data = []  # Desv. std. verosimilitud
    mean_A = []  # Media parametro A
    std_A = []  # Desv. std. parametro A
    mean_b = []  # Media parametro b
    std_b = []  # Desv. std. parametro b
    K = 1.0  # Cte. de normalizacion
    C_0 = []
    lambda_0 = []
    full_A = []  # Almacena valores de A malla
    full_b = []  # Almacena valores de b malla

    def __init__(self, var_data, mean_A, std_A, mean_b, std_b, C_0, lambda_0):
        ''' Inicia los parametros de las distribuciones a priori'''

        self.var_data = var_data
        self.mean_A = mean_A
        self.std_A = std_A
        self.mean_b = mean_b
        self.std_b = std_b
        self.C_0 = C_0
        self.lambda_0 = lambda_0

        min_A = 0.0
        max_A = mean_A * 2.0
        min_b = mean_b / 5.0
        max_b = mean_b * 5.0

        '''Divide el espacio de cada parametro en 500'''
        self.full_A = np.linspace(min_A, max_A, 500)
        self.full_b = np.linspace(min_b, max_b, 500)
    # END of __init__

    def prob_A(self, A):

        cte = 1.0 / np.sqrt(2.0 * np.pi)
        exponential = np.exp(-0.5 * ((A - self.mean_A) / self.std_A)**2)

        return cte * exponential
    # END of prob_A

    def prob_b(self, b):

        cte = 1.0 / np.sqrt(2.0 * np.pi)
        exponential = np.exp(-0.5 * ((b - self.mean_b) / self.std_b)**2)

        return cte * exponential
    # END of prob_b

    def model(self, x, A, b):

        return self.C_0 - A * np.exp(-0.5 * ((x - self.lambda_0) / b)**2)
    # END of model

    def verosimilitud(self, x, y, A, b):
        ''' Recibe vectores de datos x, y = F(x) para
        evaluar verosimilitud '''

        assert len(x) == len(y), '\
        Vectores en verosimilitud son de distinto largo'

        N = len(x)
        cte = 1 / (2 * np.pi * self.var_data) ** (N / 2.0)
        expo = np.exp(-0.5 / self.var_data *
                      np.sum(y - self.model(x, A, b))**2)
        return cte * expo
    # END of verosimilitud

    def prob_sin_norm(self, x, y, A, b):
        return self.verosimilitud(x, y, A, b) * self.prob_A(A) * self.prob_b(b)
    # END of prob_sin_norm

    def normalizar(self, x, y):

        ''' Se calcula el area total de la distribucion a posteriori
        Retorna la constante de normalizacion K'''

        delta_A = self.full_A[1] - self.full_A[0]
        delta_b = self.full_b[1] - self.full_b[0]

        K = 0.0
        for el_A in self.full_A:
            for el_b in self.full_b:
                # aux = self.prob_sin_norm(x, y, el_a, el_b)
                aux = self.verosimilitud(x, y, el_A, el_b)
                # print(aux)
                K += aux * delta_A * delta_b

        self.K = K
        return self.K
    # END of normalizar

    def prob_posteriori(self, x, y, A, b):
        ''' Probabilidad a posteriori de que parametros
        A, b se correspondan con los datos '''
        return self.prob_sin_norm(x, y, A, b) / self.K
    # END of prob_posteriori

    def prob_posteriori_A(self, x, y, A):
        ''' Retorna funcion de probabilidad de A '''

        delta_b = self.full_b[1] - self.full_b[0]

        suma = 0.0
        for el_b in self.full_b:
            suma += self.prob_posteriori(x, y, A, el_b) * delta_b

        return suma
    # END of prob_posteriori_A

    def prob_posteriori_b(self, x, y, b):
        ''' Retorna funcion de prob de b '''
        delta_A = self.full_A[1] - self.full_A[0]

        suma = 0.0
        for el_A in self.full_A:
            suma += self.prob_posteriori(x, y, el_A, b)

        return suma
    # END of prob_posteriori_b


def make_modelo1(var_data, mean_A, std_A, mean_b, std_b, C_0, lambda_0):

    ''' Permite crear un objeto para modelo1 '''
    new_system = modelo1(var_data, mean_A, std_A, mean_b, std_b, C_0, lambda_0)

    return new_system
