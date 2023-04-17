import jax
import jax.numpy as jnp
import jax.random as rand
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.messaging as msg
from biogeme.expressions import Beta
from jax.config import config

config.update("jax_debug_nans", True)
k = rand.PRNGKey(0)


def generate_data(key, n_samples, n_var, sigma):
    """

    :param key:
    :param n_samples:
    :param n_var:
    :param sigma: standard deviation of covariates.
    :return:
        x : (n_samples, n_var) array of covariates
        y: (n_samples)-length 1-D array of binary outcome variables
    """
    i = jnp.ones((n_samples, 1))
    x = sigma*rand.normal(key=key, shape=(n_samples, n_var))
    x = jnp.hstack((i, x))
    true_b = rand.uniform(key=key, shape=(n_var + 1, ), minval=-2, maxval=2)
    true_specification = jnp.sum(true_b*x, axis=1)
    prob = 1 / (1 + jnp.exp(-true_specification))  # true data generation process
    y = rand.bernoulli(key, prob)
    return x, y, true_b


def compute_choice_probability(beta, x):
    """
    use the binomial logit formula to compute individual choice probabilities from the utilities
    :param beta: current coefficient estimates
    :param x:
    :return: logit choice probabilities for each observation
    """
    utility = generate_utility(beta, x)
    return 1 / (1 + jnp.exp(-utility))


def generate_utility(beta, x):
    """
    :param beta: current coefficient estimates
    :param x: (N x m+1) array of covariates, with an intercept column
    :return: utility values for each observation
    """
    lip_specification = beta * x
    util = jnp.sum(lip_specification, axis=1)
    return util


def compute_log_likelihood(beta, x, y):
    prediction = compute_choice_probability(beta, x)
    t1 = y * prediction
    # TODO: better underflow handling of the log(1-p) calculation
    t2 = (1 - y) * (1 - prediction)
    LL = jnp.sum(jnp.log(t1 + t2))
    # print(LL)
    return LL


def compute_individual_log_likelihood(beta, x, y):
    prediction = compute_choice_probability(beta, x)
    t1 = y * jnp.log(prediction)
    t2 = (1 - y) * jnp.log(1 - prediction)
    LL = t1 + t2
    return LL


def update_params(beta, x, y):
    """
    Newton-Raphson algorithm to update the coefficients

    :param beta: current coefficient estimates
    :param x: covariates
    :param y: choice variable
    :return: updated beta values
    """
    LL = compute_log_likelihood(beta, x, y)
    gradient = jax.grad(compute_log_likelihood)(1.*beta, 1.*x, 1.*y)
    hessian = jax.hessian(compute_log_likelihood)(beta, x, y)

    invH = jnp.linalg.inv(hessian)
    g = jnp.linalg.norm(gradient)

    return beta - invH.dot(gradient), LL, g


def sample_from_data(key, X, y):
    """
    Samples (with replacement) from the original data set for bootstrapping
    :param key: key indicating random seed
    :param X: full data set of covariates
    :param y: full outcome labels
    :return: new bootstrapped data set
    """
    n_samples = X.shape[0]
    indices = rand.randint(key,
                           shape=(n_samples, ),
                           minval=0,
                           maxval=n_samples-1)

    X_boostrap = X[indices, :]
    y_bootstrap = y[indices]

    return X_boostrap, y_bootstrap


def compute_bootstrap_betas(X, y, max_iter, R):
    beta_r = jnp.zeros(shape=(R, X.shape[1]))
    k = np.arange(R)
    for r in range(R):
        key = rand.PRNGKey(k[r])
        Xb, yb = sample_from_data(key, X, y)
        print(Xb)
        init_betas = rand.uniform(key=key, shape=(X.shape[1],), minval=-5, maxval=5)  # initialize random beta values
        final_betas, ll, g, init_ll, i = estimate(key, X, y, init_betas, max_iter)
        beta_r = beta_r.at[r].set(final_betas)

    return beta_r


def compute_covariance_matrix(key, final_betas, X, y, robust=False):
    """
    there are currently errors with my implementation of the ROBUST covariance matrix
    calculation -- mainly due to numerical issues when re-estimating bootstrapped models.
    :param key:
    :param final_betas:
    :param X:
    :param y:
    :param robust:
    :return:
    """
    if robust:
        # bootstrap
        R = 100
        max_iter = 1000
        bootstrap_betas = compute_bootstrap_betas(X, y, max_iter, R)
        covs = jnp.zeros(shape=(2, 2))
        for r in range(R):
            mtrx = jnp.array([final_betas, bootstrap_betas[r]])
            covs = covs.at[:, :].add(mtrx)

        return covs/R

    else:
        # asymptotic estimate
        N = X.shape[0]
        scores = jax.jacfwd(compute_individual_log_likelihood)(1. * final_betas, 1. * X, 1. * y)
        W = jnp.cov(scores, rowvar=False)
        return jnp.linalg.inv(W) / N


def compute_standard_errors(varcov):
    var = jnp.diagonal(varcov)
    return np.sqrt(var)


def estimate(key, X, y, b, max_iter):
    n = max_iter
    g = jnp.inf
    i = 0
    betas = b
    ll = -jnp.inf
    init_ll = -jnp.inf
    while g >= 0.0001:
        if i > max_iter:
            break
        betas, ll, g = update_params(betas, X, 1 * y)
        if i == 0:
            init_ll = ll
        # print(f"({i}/{n}){':': <10}log-likelihood = {ll}")
        i += 1

    return betas, ll, g, init_ll, i


def main(key, max_iter):
    n_samples = 2000
    n_vars = 5
    X, y, true_betas = generate_data(key=key, n_samples=n_samples, n_var=n_vars, sigma=0.1)
    init_betas = rand.uniform(key=key, shape=(n_vars+1,), minval=-5, maxval=5)  # initialize random beta values

    # estimate the model
    final_betas, ll, g, init_ll, i = estimate(key, X, y, init_betas, max_iter)

    # covariance matrix
    robust = False
    cov = compute_covariance_matrix(key, final_betas, X, y, robust=robust)
    se = compute_standard_errors(cov)
    t = final_betas / se

    # estimation statistics
    print(100*"-")
    print(f"initial LL: {init_ll}\nfinal LL: {ll}")
    # print(f"mcfaddens R^2: {1 - jnp.exp(ll) / jnp.exp(init_ll)}")
    print("Newton-Raphson with norm(g) < 0.0001 convergence criterion")
    print(f"number of iterations: {i}")
    print(f"final gradient: {g}\n")
    print("coefficient estimates: \n")

    results_df = pd.DataFrame({"Variable": np.arange(n_vars + 1),
                               "B_hat": final_betas,
                               "SE": se,
                               "t": t}).set_index("Variable", drop=True)

    print(results_df)
    print(100*"-", "\n")
    print(f"estimated covariance matrix:\n{cov}")
    print(100*"-")
    # biogeme_test(key)


def biogeme_test(key):
    """
    WIP
    :param key:
    :return:
    """
    x, y, true_betas = generate_data(key, 1000, 6, 3)
    data = pd.DataFrame({"x": x, "choice": 1*y})
    database = db.Database("dummy_db", data)
    globals().update(database.variables)
    asc = Beta("asc", 0, None, None, 0)
    beta_1 = Beta("beta_1", 0, None, None, 0)
    beta_2 = Beta("beta_2", 0, None, None, 0)

    v1 = asc + beta_1 * x1 + beta_2 * x2

    V = {1: v1, 0: 0}

    loglikelihood = models.loglogit(V, None, choice)
    biogeme = bio.BIOGEME(database, loglikelihood)
    biogeme.modelName = "logit_comparison"
    results = biogeme.estimate()
    pandas_results = results.getEstimatedParameters()
    print(pandas_results)


if __name__ == "__main__":
    main(k, 1000)
