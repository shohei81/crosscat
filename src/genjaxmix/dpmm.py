from genjax import gen, repeat
from genjax import normal, inverse_gamma, dirichlet, categorical, beta
import jax.numpy as jnp
from .utils import beta_to_logpi

K = 5 
L_num = 7

@gen
def hyperparameters(mu_0=0.0, l=1.0, shape=1.0, scale=1.0, alpha=1.0):
    """
        µ, sigma^2 ~ N(µ|m_0, sigma^2*l)IG(sigma^2| shape, scale)
    """
    sigma_sq = inverse_gamma(shape*jnp.ones(K), scale*jnp.ones(K)) @ "sigma"
    sigma = jnp.sqrt(sigma_sq)
    mu = normal(mu_0 * jnp.ones(K), sigma * l) @ "mu"
    logp = dirichlet(alpha*jnp.ones((K, L_num))) @ "logp"
    logp = jnp.log(logp)
    return mu, sigma, logp

@gen
def cluster(pi, mu, sigma, logp):
    """Sample from a mixture model with proportions ``pi``, normal inverse gamma parameters ``mu`` and ``sigma``, and 
    categorical parameter ``logp``.

    Args:
        pi:
        mu:
        sigma:
        logp:
    
    Returns:
        idx:
        y1:
        y2:

    """
    idx = categorical(pi) @ "c"
    y1 = normal(mu[idx], sigma[idx]) @ "y1"
    y2 = categorical(logp[idx]) @ "y2"
    return idx, y1, y2

@gen
def gem(alpha):
    """Sample from a Griffiths, Engen, and McCloskey's (GEM) distribution with concentration ``alpha``.

    Args:
        alpha: a positive scalar
    
    Returns:
        A random array given by shape ``K``
        
    """
    betas = beta(jnp.ones(K), alpha*jnp.ones(K)) @ "pi"
    pi = beta_to_logpi(betas)
    return pi


# @gen
# def dpmm(concentration=1.0, mu_0=0.0, l=1.0, a=1.0, b=1.0):
#     """Sample from a Dirichlet process mixture model.

#     Args:
#         concentration: 
#         mu_0: ?
#         precision: ?
#         a: shape of the inverse gamma
#         b: scale of the inverse gamma
    
#     Returns:
#         A triplet ``(c, y1, y2)`` of three arrays. The first value, ``c``, is the assignments. The values ``y1` and ``y2``
#         represent the numerical and categorical features of each data point, respectively.
#     """

#     logpi = gem(concentration) @ "pi"
#     mu, sigma, logp = hyperparameters(mu_0, l, a, b) @ "hyperparameters"
#     y = cluster_repeat(logpi, mu, sigma, logp) @ "assignments"
#     return y

def generate(N_max):
    """Sample from a Dirichlet process mixture model.

    Args:
        concentration: 
        mu_0: ?
        precision: ?
        a: shape of the inverse gamma
        b: scale of the inverse gamma
    
    Returns:
        A triplet ``(c, y1, y2)`` of three arrays. The first value, ``c``, is the assignments. The values ``y1` and ``y2``
        represent the numerical and categorical features of each data point, respectively.
    """
    cluster_repeat = repeat(n=N_max)(cluster)

    @gen
    def dpmm(concentration=1.0, mu_0=0.0, l=1.0, a=1.0, b=1.0):

        logpi = gem(concentration) @ "pi"
        mu, sigma, logp = hyperparameters(mu_0, l, a, b) @ "hyperparameters"
        y = cluster_repeat(logpi, mu, sigma, logp) @ "assignments"
        return y
    return dpmm