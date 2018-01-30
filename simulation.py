"""
This module implements a simple post-estimation procedure for estimating effect
sizes after a statsmodels OLS regression. It tries to follow the CLARIFY procedure
implemented for Stata.

See:
Michael Tomz, Jason Wittenberg, and Gary King. 2003. CLARIFY: Software for
Interpreting and Presenting Statistical Results. Version 2.1. Stanford
University, University of Wisconsin, and Harvard University. January 5.
Available at http://gking.harvard.edu/

King, Tomz, and Wittenberg (2000). Making the Most of Statistical Analyses:
Improving Interpretation and Presentation. American Journal of Political
Science 44(2), 347-361.
"""

import numpy as np
import pandas as pd


def simulate(res, m=1000):
    """
    simulate takes in the results of a fitted statsmodels OLS regression and
    takes m number of draws from a multivariate normal distribution based
    on its point estimates and variance-covariance matrix. It then simulates
    the ancillary parameter sigma squared. It returns a pandas.DataFrame with
    the simulated parameters.
    """
    # simulate main parameters
    betas = res.params.as_matrix()
    var_cov = res.cov_params().as_matrix()
    parameters = np.random.multivariate_normal(betas, var_cov, size=m)
    sim_results = pd.DataFrame(parameters, columns=res.params.index)

    # simulate sigma squared
    mse = res.mse_resid
    df = res.df_resid
    sigmas = df * mse / np.random.chisquare(df, size=m)
    sim_results['sigma_squared'] = sigmas

    return sim_results


def sim_predict(sim_results, setx, quantity='expected'):
    pred = sim_results.copy()
    for x, value in setx.items():
        pred[x] = pred.loc[:, x] * value

    if quantity == 'predicted':
        pred['e'] = np.random.normal(0, np.sqrt(pred['sigma_squared']))

    return pred.loc[:, pred.columns != 'sigma_squared'].sum(axis=1)
