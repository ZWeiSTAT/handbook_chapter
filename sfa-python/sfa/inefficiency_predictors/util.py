import numpy as np
import pandas as pd

from scipy import stats


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