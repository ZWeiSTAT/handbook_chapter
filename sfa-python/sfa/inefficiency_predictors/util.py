import numpy as np
import pandas as pd

from scipy import stats
from scipy.linalg import expm, logm, norm


def JLMS_panel_technical_inefficiency_scores(theta, y, X):
    N = 171
    T = 6

    alpha = theta[0]
    beta = theta[1:14]
    sigma = theta[-2]
    _lambda = theta[-1]

    u_hat = np.zeros((N, T))
    for t in range(T):

        eps_t = y[t] - alpha - X[t] @ beta
        b = (eps_t * _lambda) / sigma
        u_hat[:, t] = ((sigma * _lambda) / (1 + _lambda**2)) * (
            stats.norm.pdf(b) / (1 - stats.norm.cdf(b)) - b
        )

    return u_hat


def inverse_mapping_vec(gamma, tol_value=1e-8):
    C = []
    iter_number = -1

    if not isinstance(gamma, (np.ndarray, list)):
        raise ValueError
    if isinstance(gamma, np.ndarray):
        if gamma.ndim != 1:
            raise ValueError
    n = 0.5 * (1 + np.sqrt(1 + 8 * len(gamma)))
    if not all([n.is_integer(), n > 1]):
        raise ValueError

    if not (0 < tol_value <= 1e-4):
        tol_value = 1e-8
        print("Warning: tolerance value has been changed to default")

    n = int(n)
    A = np.zeros(shape=(n, n))
    A[np.triu_indices(n, 1)] = gamma
    A = A + A.T

    diag_vec = np.diag(A)
    diag_ind = np.diag_indices_from(A)

    dist = np.sqrt(n)
    while dist > np.sqrt(n) * tol_value:
        diag_delta = np.log(np.diag(expm(A)))
        diag_vec = diag_vec - diag_delta
        A[diag_ind] = diag_vec
        dist = norm(diag_delta)
        iter_number += 1

    C = expm(A)
    np.fill_diagonal(C, 1)

    return C


def NW_panel_technical_inefficiency_scores(theta, y, X, U_hat):
    N = 171
    T = 6

    alpha = theta[0]
    beta = theta[1:14]
    sigma = theta[-2]
    _lambda = theta[-1]

    sigma2u = ((_lambda * sigma) / (1 + _lambda)) ** 2
    sigma2v = (sigma / (1 + _lambda)) ** 2

    obs_eps = np.zeros((N, T))
    for t in range(T):
        obs_eps[:, t] = y[t] - alpha - X[t] @ beta

    simulated_u = stats.halfnorm.ppf(
        U_hat, loc=np.zeros(T), scale=np.array([np.sqrt(sigma2u) for x in range(T)])
    )
    simulated_v = stats.multivariate_normal.rvs(
        size=10000, mean=np.zeros(T), cov=np.eye(T) * sigma2v, random_state=123
    )
    simulated_eps = simulated_v - simulated_u

    h_eps = (
        1.06
        * 10000 ** (-1 / 5)
        * (max(np.std(simulated_eps), stats.iqr(simulated_eps) / 1.34))
    )

    u_hat = np.zeros((N, T))
    for i in range(N):
        panel_i_kernel_regression_results = np.zeros(T)
        eps_kernel = np.zeros((10000, T))
        for t in range(T):
            eps_kernel[:, t] = stats.norm.pdf(
                (simulated_eps[:, t] - obs_eps[i, t]) / h_eps
            )
        kernel_product = np.prod(eps_kernel, 1)
        for j in range(T):
            panel_i_kernel_regression_results[j] = np.sum(
                kernel_product * simulated_u[:, j]
            ) / np.sum(kernel_product)
        u_hat[i, :] = panel_i_kernel_regression_results

    return u_hat