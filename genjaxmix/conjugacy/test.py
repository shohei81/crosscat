from sufficient_statistics import segment_sufficient_statistic as ssa
import genjaxmix.distributions as dist
import jax
import jax.numpy as jnp

ss = ssa(dist.NormalInverseGamma(), dist.Normal())
x = jnp.array([1.0,2,3])
assignments = jnp.array([0,0,1], dtype=jax.numpy.int32)
a_0 = jnp.ones(2)
b_0 = jnp.ones(2)
mu_0 = jnp.zeros(2)
v_0 = jnp.ones(2)

a, b, c, d = ss((a_0, b_0, mu_0, v_0), x, assignments)

def compile(parameters, x, assignments, dist1, dist2):
    ss = ssa(dist1, dist2)
    return ss(parameters, x, assignments)

jitted = jax.jit(compile, static_argnames=("dist1", "dist2"))

x = jnp.array([0,1,1,0], dtype=jax.numpy.int32)
assignments = jnp.array([0,0,1,1], dtype=jax.numpy.int32)
jitted((a_0, b_0), x, assignments, dist.Beta(), dist.Bernoulli())