from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from IPython.display import Image
from scipy.optimize import leastsq

plt.style.use('bmh')
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['figure.figsize'] = '7, 5'


def leer_archivo(nombre):
    '''
    lee el archivo
    nombre debe ser un str
    '''
    datos = np.loadtxt(nombre)
    x = np.array(datos[:, 0])
    y = np.array(datos[:, 1])
    return x, y


def modelo_1(p, x):
    A, sigma = p
    a = 10 ** (- 1)
    mu = 6563
    y = a - A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return y


def modelo_2(p, x):
    A1, sigma1, A2, sigma2 = p
    a = 10 ** (- 1)
    mu = 6563
    y = a - A1 * scipy.stats.norm(loc=mu, scale=sigma1).pdf(x) - A2 * scipy.stats.norm(loc=mu, scale=sigma2).pdf(x)
    return y


def gauss2d(x, y, mat_sigma):
    sigma_x, sigma_y, rho = mat_sigma
    A = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))
    B = (- 1 / (2 * (2 - rho ** 2))) * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2
                                        - 2 * rho * x * y / (sigma_x * sigma_y))
    return A * np.exp(B)


def prior(beta, p, model=2):
    if model == 2:
        beta0, beta1, beta2, beta3 = beta
        a0, a1, a2, a3, b0, b1, b2, b3 = p
        s0 = ((beta0 - a0) / b0) ** 2
        s1 = ((beta1 - a1) / b1) ** 2
        s2 = ((beta2 - a2) / b2) ** 2
        s3 = ((beta3 - a3) / b3) ** 2
        s = (s0 + s1 + s2 + s3) / 2.
        P = np.exp(-s) / (4 * np.pi ** 2 * b0 * b1 * b2 * b3)
        return P
    elif model == 1:
        beta0, beta1 = beta
        a0, a1, b0, b1 = p
        s0 = ((beta0 - a0) / b0) ** 2
        s1 = ((beta1 - a1) / b1) ** 2
        s = (s0 + s1) / 2.
        P = np.exp(-s) / (2 * np.pi * b0 * b1)
        return P


def fill_prior(beta_grid, prior_p, model=2):
    if model == 2:
        beta0_grid, beta1_grid, beta2_grid, beta3_grid = beta_grid
        salida = np.zeros(beta0_grid.shape)
        ni, nj, nk, nl = beta0_grid.shape
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        salida[i, j, k, l] = prior([beta0_grid[i, j, k, l], beta1_grid[i,j, k, l], beta2_grid[i, j, k, l], beta3_grid[i,j, k, l]], prior_p, 2)
    elif model == 1:
        beta0_grid, beta1_grid = beta_grid
        salida = np.zeros(beta0_grid.shape)
        ni, nj = beta0_grid.shape
        for i in range(ni):
            for j in range(nj):
                salida[i, j] = prior([beta0_grid[i, j], beta1_grid[i,j]], prior_p, 1)
    return salida


def likelihood(beta, datos, error, model=2):
    x, y = datos
    N = len(x)
    if model == 2:
        s = np.sum(y - modelo_2(beta, x))
        L = (2 * np.pi * error ** 2) ** (-N / 2.) * np.exp(- s / (2 * error ** 2))
    elif model == 1:
        s = np.sum(y - modelo_1(beta, x))
        L = (2 * np.pi * error ** 2) ** (-N / 2) * np.exp(-s / (2 * error ** 2))
    return L


def fill_likelihood(beta_grid, datos, error, model=2):
    if model == 2:
        beta0_grid, beta1_grid, beta2_grid, beta3_grid = beta_grid
        salida = np.zeros(beta0_grid.shape)
        ni, nj, nk, nl = beta0_grid.shape
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        salida[i, j, k, l] = likelihood([beta0_grid[i, j, k, l], beta1_grid[i,j, k, l], beta2_grid[i, j, k, l], beta3_grid[i,j, k, l]], datos, error, 2)
    elif model == 1:
        beta0_grid, beta1_grid = beta_grid
        salida = np.zeros(beta0_grid.shape)
        ni, nj = beta0_grid.shape
        for i in range(ni):
            for j in range(nj):
                salida[i, j] = likelihood([beta0_grid[i, j], beta1_grid[i,j]], datos, error, 1)
    return salida


def chi_cuad(f, x, y, p):
    n = len(x)
    chi = 0
    for i in range(n):
        chi = chi + (y[i] - f(x[i], p))
    return chi


def make_figure_axes(x, y, fig_number=1, fig_size=8):
    '''
    Creates a set of 3 axes to plot 2D function + marginals
    '''
    # determine max size
    size_x = x.max() - x.min()
    size_y = y.max() - y.min()
    max_size = max(size_x, size_y)
    min_size = min(size_x, size_y)

    if size_x >= size_y:
        fig_size_x = fig_size
        fig_size_y = (0.12 * fig_size_x +
                      0.65 * fig_size_x * min_size / max_size +
                      0.02 * fig_size_x +
                      0.18 * fig_size_x +
                      0.03 * fig_size_x)
        rect_main = [0.12,
                     0.12 * fig_size_x / fig_size_y,
                     0.65,
                     0.65 * fig_size_x * min_size / max_size / fig_size_y]
        rect_x = [0.12, ((0.12 + 0.65 * min_size / max_size + 0.02) *
                         fig_size_x / fig_size_y),
                  0.65, 0.18 * fig_size_x / fig_size_y]
        rect_y = [0.79, 0.12 * fig_size_x / fig_size_y,
                  0.18, 0.65 * fig_size_x * min_size / max_size / fig_size_y]

    else:
        fig_size_y = fig_size
        fig_size_x = (0.12 * fig_size_y +
                      0.65 * fig_size_y * min_size / max_size +
                      0.02 * fig_size_y +
                      0.18 * fig_size_y +
                      0.03 * fig_size_y)
        rect_main = [0.12 * fig_size_y / fig_size_x,
                     0.12,
                     0.65 * fig_size_y * min_size / max_size / fig_size_x,
                     0.65]
        rect_x = [0.12 * fig_size_y / fig_size_x, 0.79,
                  0.65 * fig_size_y * min_size / max_size / fig_size_x, 0.18]
        rect_y = [((0.12 + 0.65 * min_size / max_size + 0.02) *
                   fig_size_y / fig_size_x), 0.12,
                  0.18 * fig_size_y / fig_size_x, 0.65]

    fig = plt.figure(fig_number, figsize=(fig_size_x, fig_size_y))
    fig.clf()
    ax_main = fig.add_axes(rect_main)
    ax_marginal_x = fig.add_axes(rect_x, xticklabels=[])
    ax_marginal_y = fig.add_axes(rect_y, yticklabels=[])

    return ax_main, ax_marginal_x, ax_marginal_y


def plot_distribution(x, y, z, cmap='PuBu_r'):
    x_limits = (x.min(), x.max())
    y_limits = (y.min(), y.max())

    ax_main, ax_marginal_x, ax_marginal_y = make_figure_axes(x, y)
    ax_main.pcolormesh(x, y, z, cmap=cmap)

    marginal_x = np.sum(z, axis=1)
    ax_marginal_x.plot(x[:, 0], marginal_x)
    [l.set_rotation(-90) for l in ax_marginal_y.get_xticklabels()]

    marginal_y = np.sum(z, axis=0)
    ax_marginal_y.plot(marginal_y, y[0])

    ax_main.set_xlim(x_limits)
    ax_main.set_ylim(y_limits)

    ax_marginal_x.set_xlim(x_limits)
    ax_marginal_y.set_ylim(y_limits)
    return ax_main, ax_marginal_x, ax_marginal_y


def marginal1(grid, dx, dy):
    P_E = np.sum(grid) * dx * dy
    A = np.sum(grid, axis=1) * dy / P_E
    std = np.sum(grid, axis=1) * dx / P_E
    return A, std


def evidencia1(grid, dx, dy):
    P_E = np.sum(grid) * dx * dy
    A = np.sum(grid, axis=1) * dx / P_E
    std = np.sum(grid, axis=1) * dy / P_E
    E_A = np.sum(grid[:, 0] * A) * dx
    E_std = np.sum(grid[0, :] * std) * dy
    return A, std, E_A, E_std, P_E


def residuo_1(p, x_exp, y_exp):
    er = y_exp - modelo_1(p, x_exp)
    return er


def err(x_exp, y_exp, p0):
    aprox = leastsq(residuo_1, p0, args=(x_exp, y_exp))
    return aprox


def evidencia2(grid, dx2, dy2, dx3, dy3):
    P_E = np.sum(grid) * dx2 * dy2 * dx3 * dy3
    A_1 = (np.sum(np.sum(np.sum(grid, axis=1), axis=1), axis=1) * dy2 * dx3 * dy3 / P_E)
    std_1 = (np.sum(np.sum(np.sum(grid, axis=1), axis=1), axis=1) * dx2 * dx3 * dy3 / P_E)
    A_2 = (np.sum(np.sum(np.sum(grid, axis=1), axis=1), axis=1) * dx2 * dy2 * dy3 / P_E)
    std_2 = (np.sum(np.sum(np.sum(grid, axis=1), axis=1), axis=1) * dx2 * dy2 * dx3 / P_E)
    E_A_1 = np.sum(grid[:, 0, 0, 0] * A_1) * dx2
    E_std_1 = np.sum(grid[:, 0, 0, 0] * std_1) * dy2
    E_A_2 = np.sum(grid[:, 0, 0, 0] * A_2) * dx3
    E_std_2 = np.sum(grid[:, 0, 0, 0] * std_2) * dy3
    return A_1, std_1, A_2, std_2, E_A_1, E_std_1, E_A_2, E_std_2, P_E


# main
x, y = leer_archivo('espectro.dat')
datos = (x, y * 10 ** 15)
# modelo 1
x_exp = x
y_exp = y * 10 ** 15
p0 = (-70., 5.)
e = err(x_exp, y_exp, p0)
print (e)
error = 1
beta_grid1 = np.mgrid[-100.:-40.:50j, 2.:10.:50j]
beta0_grid, beta1_grid = beta_grid1
ad1 = [-70., 5., 7, 4]
a = fill_prior(beta_grid1, ad1, 1)
b = fill_likelihood(beta_grid1, datos, error, 1)
c = a * b
dx1 = 60. / 50.
dy1 = 8. / 50.
marginal_A, marg_std, E_A, E_std, P_E_1 = evidencia1(c, dx1, dy1)
plot_distribution(beta0_grid, beta1_grid, c)
plt.show()
# modelo 2
beta_grid2 = np.mgrid[-80:-20:50j, 2:20:50j, -60:-5:50j, 2:25:50j]
ad2 = [50, 40, 10, 15, 30, 30, 10, 10]
d = fill_prior(beta_grid2, ad2, 2)
e = fill_likelihood(beta_grid2, datos, error, 2)
f = d * e
dx2 = 60. / 50.
dy2 = 18. / 50.
dx3 = 55. / 50.
dy3 = 23. / 50.
marginal_A_1, marg_std_1, marginal_A_2, marg_std_2, E_A_1, E_std_1, E_A_2, E_std_2, P_E_2 = evidencia2(f, dx2, dy2, dx3, dy3)
print (P_E_1 / P_E_2)
