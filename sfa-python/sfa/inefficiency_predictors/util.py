import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

from scipy import stats
from scipy.linalg import expm, logm, norm
from scipy.optimize import minimize


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


def direct_mapping_mat(C):
    gamma = []

    try:
        if not isinstance(C, np.ndarray):
            raise ValueError
        if C.ndim != 2:
            raise ValueError
        if C.shape[0] != C.shape[1]:
            raise ValueError
        if not all(
            [
                np.all(np.abs(np.diag(C) - np.ones(C.shape[0])) < 1e-8),
                np.all(np.linalg.eigvals(C) > 0),
                np.all(np.abs(C - C.T) < 1e-8),
            ]
        ):
            raise ValueError

        A = logm(C)
        gamma = A[np.triu_indices(C.shape[0], 1)]

    except ValueError:
        print("Error: input is of a wrong format")

    return gamma


def Loglikelihood_Gaussian_copula_cross_sectional_application_SFA(
    theta, y, X, P, us_Sxn, n_inputs, S
):
    n = len(y)

    alpha = np.exp(theta[0])
    beta = np.exp(theta[1 : n_inputs + 1])
    sigma2_v = theta[1 + n_inputs]
    sigma2_u = theta[2 + n_inputs]
    sigma2_w = np.exp(theta[3 + n_inputs : (3 + n_inputs) + (n_inputs - 1)])
    mu_W = theta[
        (3 + n_inputs) + (n_inputs - 2) + 1 : (3 + n_inputs) + (n_inputs - 2) + (n_inputs)
    ]

    rhos_log_form = theta[-3:]

    Rho_hat = inverse_mapping_vec(rhos_log_form)

    eps = y - np.log(alpha) - X @ beta
    W = (np.tile(X[:, 0].reshape(-1, 1), (1, n_inputs - 1)) - X[:, 1:]) - (
        P[:, 1:]
        - np.tile(P[:, 0].reshape(-1, 1), (1, n_inputs - 1))
        + (np.log(beta[0]) - np.log(beta[1:]))
    )
    Den_W = stats.norm.pdf(
        W, np.tile(mu_W, (n, 1)), np.tile(np.sqrt(sigma2_w), (n, 1))
    )
    CDF_W = stats.norm.cdf(
        W, np.tile(mu_W, (n, 1)), np.tile(np.sqrt(sigma2_w), (n, 1))
    )

    eps_Sxn = np.repeat(eps.reshape(-1, 1), S, axis=1).T
    us_Sxn_scaled = np.sqrt(sigma2_u) * us_Sxn
    CdfUs = 2 * (stats.norm.cdf(np.sqrt(sigma2_u) * us_Sxn, 0, np.sqrt(sigma2_u)) - 0.5)
    eps_plus_us = eps_Sxn + us_Sxn_scaled
    den_eps_plus_us = stats.norm.pdf(eps_plus_us, 0, np.sqrt(sigma2_v))

    simulated_copula_pdfs = np.zeros((S, n))
    CDF_W_rep = {}
    for i in range(n_inputs - 1):
        CDF_W_rep[i] = np.repeat(CDF_W[:, i].reshape(-1, 1), S, axis=1).T

    for j in range(n):
        CDF_w_j = np.zeros((S, n_inputs - 1))
        for i in range(n_inputs - 1):
            CDF_w_j[:, i] = CDF_W_rep[i][:, j]
        U = np.concatenate(
            [stats.norm.ppf(CdfUs[:, j]).reshape(-1, 1), stats.norm.ppf(CDF_w_j)], axis=1
        )
        c123 = stats.multivariate_normal.pdf(
            U, mean=np.array([0, 0, 0]), cov=Rho_hat
        ) / np.prod(stats.norm.pdf(U), axis=1)
        simulated_copula_pdfs[:, j] = c123

    Integral = np.mean(simulated_copula_pdfs * den_eps_plus_us, axis=0)
    prod_Den_W = np.prod(Den_W, 1)
    DenAll = prod_Den_W * Integral
    DenAll[DenAll < 1e-6] = 1e-6
    r = np.log(np.sum(beta))
    logDen = np.log(DenAll) + r
    logL = -np.sum(logDen)

    return logL


def estimate_Jondrow1982_u_hat(theta, n_inputs, n_corr_terms, y, X):
    alpha = theta[0]
    beta = theta[1 : n_inputs + 1]
    sigma2_v = theta[1 + n_inputs]
    sigma2_u = theta[2 + n_inputs]

    obs_eps = y - np.log(alpha) - X @ beta
    _lambda = np.sqrt(sigma2_u / sigma2_v)
    sigma = np.sqrt(sigma2_u + sigma2_v)
    sig_star = np.sqrt(sigma2_u * sigma2_v / (sigma**2))
    u_hat = sig_star * (
        (stats.norm.pdf(_lambda * obs_eps / sigma))
        / (1 - stats.norm.cdf(_lambda * obs_eps / sigma))
        - ((_lambda * obs_eps) / sigma)
    )
    V_u_hat = sig_star**2 * (
        1
        + stats.norm.pdf(_lambda * obs_eps / sigma)
        / (1 - stats.norm.cdf(_lambda * obs_eps / sigma))
        * _lambda
        * obs_eps
        / sigma
        - (
            stats.norm.pdf(_lambda * obs_eps / sigma)
            / (1 - stats.norm.cdf(_lambda * obs_eps / sigma))
        )
        ** 2
    )

    return u_hat, V_u_hat


def Estimate_Jondrow1982_u_hat_panel_SFA_application_RS2007(
    alpha, beta, delta, sigma2_v, y, X, T, N
):
    obs_eps = {}
    u_hat = np.zeros((N, T))
    V_u_hat = np.zeros((N, T))
    for t in range(T):
        sigma2_u = np.exp(delta[0] + delta[1] * t)
        _lambda = np.sqrt(sigma2_u) / np.sqrt(sigma2_v)
        sigma = np.sqrt(sigma2_u + sigma2_v)
        sig_star = np.sqrt(sigma2_u * sigma2_v / (sigma**2))

        u_hat_ = np.full(N, np.nan)
        V_u_hat_ = np.full(N, np.nan)
        obs_eps[t] = y[t] - np.log(alpha) - X[t] @ beta
        b = (obs_eps[t] * _lambda) / sigma
        u_hat_tmp = ((sigma * _lambda) / (1 + _lambda**2)) * (
            stats.norm.pdf(b) / (1 - stats.norm.cdf(b)) - b
        )
        V_u_hat_tmp = sig_star**2 * (
            1
            + stats.norm.pdf(b) / (1 - stats.norm.cdf(b)) * b
            - (stats.norm.pdf(b) / (1 - stats.norm.cdf(b))) ** 2
        )

        u_hat_[: len(u_hat_tmp)] = u_hat_tmp
        V_u_hat_[: len(V_u_hat_tmp)] = V_u_hat_tmp
        u_hat[:, t] = u_hat_
        V_u_hat[:, t] = V_u_hat_

    return u_hat, V_u_hat


def simulate_error_components(Rho, n_inputs, S_kernel, seed):
    chol_of_rho = np.linalg.cholesky(Rho)
    Z = stats.multivariate_normal.rvs(
        mean=np.zeros(n_inputs), cov=np.eye(n_inputs), size=S_kernel, random_state=seed
    )
    X = chol_of_rho @ Z.T
    U = stats.norm.cdf(X)

    return U.T


def Estimate_NW_u_hat_conditional_eps_panel_SFA_RS2007(
    theta, y, X, N, T, k, U_hat, S_kernel
):
    alpha = theta[0]
    beta = theta[1 : k + 1]
    delta = theta[4:6]
    sigma2_v = theta[6]

    obs_eps = {}
    for t in range(T):
        eps__ = np.full(N, np.nan)
        tmp_eps = y[t] - np.log(alpha) - X[t] @ beta
        eps__[: len(tmp_eps)] = tmp_eps
        obs_eps[t] = eps__

    simulated_v = stats.multivariate_normal.rvs(
        mean=np.zeros(T), cov=np.eye(T) * sigma2_v, size=S_kernel
    )
    simulated_u = np.zeros((S_kernel, T))
    simulated_eps = np.zeros((S_kernel, T))
    for t in range(T):
        sigma2_u = np.exp(delta[0] + delta[1] * t)
        simulated_u[:, t] = np.sqrt(sigma2_u) * stats.norm.ppf((U_hat[:, t] + 1) / 2)
        simulated_eps[:, t] = simulated_v[:, t] - simulated_u[:, t]

    h_eps = np.zeros(T)
    for t in range(T):
        h_eps[t] = 1.06 * S_kernel ** (-1 / 5) * (
            max(np.std(simulated_eps[:, t]), stats.iqr(simulated_eps[:, t]) / 1.34)
        )

    u_hat = np.zeros((N, T))
    u_hat2 = np.zeros((N, T))
    all_eps = np.concatenate([x.reshape(-1, 1) for x in obs_eps.values()], axis=1)
    for i in range(N):
        obs_eps_i = all_eps[i, :]

        panel_i_kernel_regression_results = np.zeros(T)
        panel_i_kernel_regression_results2 = np.zeros(T)
        eps_kernel = np.zeros((S_kernel, T))
        for t in range(T):
            eps_kernel[:, t] = stats.norm.pdf(
                (simulated_eps[:, t] - obs_eps[t][i]) / h_eps[t]
            )

        out = eps_kernel[:, np.all(~np.isnan(eps_kernel), axis=0)]
        kernel_product = np.prod(out, 1)
        for j in range(T):
            if not np.isnan(obs_eps_i[j]):
                panel_i_kernel_regression_results[j] = np.sum(
                    kernel_product * simulated_u[:, j]
                ) / np.sum(kernel_product)
                panel_i_kernel_regression_results2[j] = np.sum(
                    kernel_product * simulated_u[:, j] ** 2
                ) / np.sum(kernel_product)
            else:
                panel_i_kernel_regression_results[j] = np.nan
                panel_i_kernel_regression_results2[j] = np.nan

        u_hat[i, :] = panel_i_kernel_regression_results
        u_hat2[i, :] = panel_i_kernel_regression_results2

    V_u_hat = u_hat2 - (u_hat**2)

    return u_hat, V_u_hat


def Estimate_NW_u_hat_conditional_W_cross_sectional_application(
    theta, n_inputs, n_corr_terms, y, X, P, U_hat, S_kernel
):
    n = len(y)

    alpha = theta[0]
    beta = theta[1 : n_inputs + 1]
    sigma2_v = theta[1 + n_inputs]
    sigma2_u = theta[2 + n_inputs]
    sigma2_w = np.exp(theta[3 + n_inputs : (3 + n_inputs) + (n_inputs - 1)])
    mu_W = theta[
        (3 + n_inputs) + (n_inputs - 2) + 1 : (3 + n_inputs) + (n_inputs - 2) + (n_inputs)
    ]

    obs_eps = y - np.log(alpha) - X @ beta
    W = (np.tile(X[:, 0].reshape(-1, 1), (1, n_inputs - 1)) - X[:, 1:]) - (
        P[:, 1:]
        - np.tile(P[:, 0].reshape(-1, 1), (1, n_inputs - 1))
        + (np.log(beta[0]) - np.log(beta[1:]))
    )
    rep_obs_eps = np.repeat(obs_eps.reshape(-1, 1), S_kernel, axis=1).T
    rep_obs_W = {}
    for i in range(n_inputs - 1):
        rep_obs_W[i] = np.repeat(W[:, i].reshape(-1, 1), S_kernel, axis=1).T

    simulated_v = stats.norm.rvs(loc=0, scale=np.sqrt(sigma2_v), size=S_kernel)
    simulated_u = np.sqrt(sigma2_u) * stats.norm.ppf((U_hat[:, 0] + 1) / 2)
    simulated_W = np.zeros((S_kernel, n_inputs - 1))
    for i in range(n_inputs - 1):
        simulated_W[:, i] = stats.norm.ppf(U_hat[:, i + 1], mu_W[i], np.sqrt(sigma2_w[i]))
    simulated_eps = simulated_v - simulated_u

    h_eps = 1.06 * S_kernel ** (-1 / 5) * (
        max(np.std(simulated_eps), stats.iqr(simulated_eps) / 1.34)
    )
    h_W = np.zeros(n_inputs - 1)
    for i in range(n_inputs - 1):
        h_W[i] = 1.06 * S_kernel ** (-1 / 5) * (
            max(np.std(simulated_W[:, i]), stats.iqr(simulated_W[:, i]) / 1.34)
        )
    h = np.concatenate([np.array([h_eps]), h_W])

    kernel_regression_results1 = np.zeros(n)
    kernel_regression_results2 = np.zeros(n)
    for i in range(n):
        eps_kernel = stats.norm.pdf((simulated_eps - rep_obs_eps[:, i]) / h[0])
        W_kernel = np.zeros((S_kernel, n_inputs - 1))
        for j in range(n_inputs - 1):
            W_kernel[:, j] = stats.norm.pdf((simulated_W[:, j] - rep_obs_W[j][:, i]) / h[j + 1])

        W_kernel_prod = np.prod(W_kernel, 1)
        kernel_product = eps_kernel * W_kernel_prod
        kernel_regression_results1[i] = np.sum(kernel_product * simulated_u) / np.sum(
            kernel_product
        )
        kernel_regression_results2[i] = np.sum(kernel_product * (simulated_u**2)) / np.sum(
            kernel_product
        )

    u_hat = kernel_regression_results1
    V_u_hat = kernel_regression_results2 - (u_hat**2)

    return u_hat, V_u_hat


def Loglikelihood_APS14_dynamic_panel_SFA_u_RS2007(theta, y, X, N, T, k, S, FMSLE_us):
    if np.any(np.isnan(theta)):
        logDen = np.ones((N, 1)) * -1e8
        logL = -np.sum(logDen)
    else:
        rhos = theta[7:]
        Rho = inverse_mapping_vec(rhos)

        alpha = np.exp(theta[0])
        beta = 1 / (1 + np.exp(-theta[1:4]))
        delta = theta[4:6]
        sigma2_v = theta[6]

        eps = {}
        for t in range(T):
            eps__ = np.full(N, np.nan)
            tmp_eps = y[t] - np.log(alpha) - X[t] @ beta
            eps__[: len(tmp_eps)] = tmp_eps
            eps[t] = eps__

        if not np.all(np.linalg.eigvals(Rho) > 0):
            logDen = np.ones((N, 1)) * -1e8
            logL = -np.sum(logDen)
        else:
            A = np.linalg.cholesky(Rho)
            all_eps = np.concatenate([_eps.reshape(-1, 1) for _eps in eps.values()], axis=1)
            FMSLE_densities = np.zeros(N)
            for i in range(N):
                eps_i = all_eps[i, :]
                n_NaNs = len(eps_i[np.isnan(eps_i)])
                eps_i = eps_i[~np.isnan(eps_i)]
                rep_eps_i = np.tile(eps_i, (S, 1))
                simulated_us_i = np.zeros((S, T))
                for t in range(T):
                    simulated_us_i[:, t] = FMSLE_us[t][:, i]

                CDF_u_i = stats.norm.cdf(simulated_us_i @ A, np.zeros((S, T)), np.ones((S, T)))
                sigma2_u_hat = np.exp(delta[0] + delta[1] * np.arange(1, T + 1, 1))
                u_i = stats.halfnorm.ppf(
                    CDF_u_i,
                    np.zeros((S, T)),
                    np.tile(np.ones((1, T)) * np.sqrt(sigma2_u_hat), (S, 1)),
                )

                u_i = u_i[:, : T - n_NaNs]
                den_i = np.mean(
                    stats.multivariate_normal.pdf(
                        rep_eps_i + u_i, mean=np.zeros(T - n_NaNs), cov=np.eye(T - n_NaNs) * sigma2_v
                    )
                )
                if den_i < 1e-10:
                    den_i = 1e-10
                FMSLE_densities[i] = den_i

            logL = -np.sum(np.log(FMSLE_densities))

    return logL


def export_simulation_data_RS2007_electricity_application(
    theta, n_inputs, y, X, P, U_hat, S_kernel
):
    alpha = theta[0]
    beta = theta[1 : n_inputs + 1]
    sigma2_v = theta[1 + n_inputs]
    sigma2_u = theta[2 + n_inputs]
    sigma2_w = np.exp(theta[3 + n_inputs : (3 + n_inputs) + (n_inputs - 1)])
    mu_W = theta[
        (3 + n_inputs) + (n_inputs - 2) + 1 : (3 + n_inputs) + (n_inputs - 2) + (n_inputs)
    ]

    obs_eps = y - np.log(alpha) - X @ beta
    W = (np.tile(X[:, 0].reshape(-1, 1), (1, n_inputs - 1)) - X[:, 1:]) - (
        P[:, 1:]
        - np.tile(P[:, 0].reshape(-1, 1), (1, n_inputs - 1))
        + (np.log(beta[0]) - np.log(beta[1:]))
    )
    rep_obs_W = {}
    for i in range(n_inputs - 1):
        rep_obs_W[i] = np.repeat(W[:, i].reshape(-1, 1), S_kernel, axis=1).T

    simulated_v = stats.norm.rvs(loc=0, scale=np.sqrt(sigma2_v), size=S_kernel)
    simulated_u = np.sqrt(sigma2_u) * stats.norm.ppf((U_hat[:, 0] + 1) / 2)
    simulated_W = np.zeros((S_kernel, n_inputs - 1))
    for i in range(n_inputs - 1):
        simulated_W[:, i] = stats.norm.ppf(U_hat[:, i + 1], mu_W[i], np.sqrt(sigma2_w[i]))
    simulated_eps = simulated_v - simulated_u

    NN_train_eps_W = pd.DataFrame(
        np.concatenate([simulated_eps.reshape(-1, 1), simulated_W], axis=1),
        columns=["train_simulated_eps"] + [f"train_simulated_w{i + 1}" for i in range(n_inputs - 1)],
    )
    NN_train_u = pd.DataFrame(simulated_u, columns=["train_simulated_u"])
    NN_test_eps_W = pd.DataFrame(
        np.concatenate([obs_eps.reshape(-1, 1), W], axis=1),
        columns=["test_simulated_eps"] + [f"test_simulated_w{i + 1}" for i in range(n_inputs - 1)],
    )

    NN_train_eps_W.to_csv(
        r"./cross_sectional_SFA_RS2007_electricity_application_NN_train_eps_W.csv",
        index=False,
    )
    NN_train_u.to_csv(
        r"./cross_sectional_SFA_RS2007_electricity_application_NN_train_u.csv",
        index=False,
    )
    NN_test_eps_W.to_csv(
        r"./cross_sectional_SFA_RS2007_electricity_application_NN_test_eps_W.csv",
        index=False,
    )


def export_RS2007_electricity_SFA_panel_data(theta, y, X, N, T, U_hat, S_kernel):
    alpha = theta[0]
    beta = theta[1:4]
    delta = theta[4:6]
    sigma2_v = theta[6]

    obs_eps = {}
    for t in range(T):
        eps__ = np.full(N, np.nan)
        tmp_eps = y[t] - np.log(alpha) - X[t] @ beta
        eps__[: len(tmp_eps)] = tmp_eps
        obs_eps[t] = eps__

    obs_eps = np.concatenate([x.reshape(-1, 1) for x in obs_eps.values()], axis=1)

    simulated_v = stats.multivariate_normal.rvs(np.zeros(T), np.eye(T) * sigma2_v, S_kernel)
    simulated_u = np.zeros((S_kernel, T))
    simulated_eps = np.zeros((S_kernel, T))
    for t in range(T):
        sigma2_u = np.exp(delta[0] + delta[1] * t)
        simulated_u[:, t] = np.sqrt(sigma2_u) * stats.norm.ppf((U_hat[:, t] + 1) / 2)
        simulated_eps[:, t] = simulated_v[:, t] - simulated_u[:, t]

    NN_train_eps = pd.DataFrame(simulated_eps, columns=[f"train_eps_{t}" for t in range(T)])
    NN_train_u = pd.DataFrame(simulated_u, columns=[f"train_u_{t}" for t in range(T)])
    NN_test_eps = pd.DataFrame(obs_eps, columns=[f"test_eps_{t}" for t in range(T)])

    NN_train_eps.to_csv(r"./panel_SFA_RS2007_electricty_application_NN_train_eps.csv", index=False)
    NN_train_u.to_csv(r"./panel_SFA_RS2007_electricty_application_NN_train_u.csv", index=False)
    NN_test_eps.to_csv(r"./panel_SFA_RS2007_electricty_application_NN_test_eps.csv", index=False)