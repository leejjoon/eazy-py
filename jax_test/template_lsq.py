import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
# import qpax
import timeit

from jaxopt import BoxCDQP

# from test_jaxopt import nnls
def nnls(F, g):
    nband, ntmpl = F.shape

    Q = F.T @ F
    q = -F.T @ g


    # c = jnp.array(fnu_i) # zeros(nband)#array([1.0, -1.0])
    l = jnp.zeros(ntmpl)#array([1.0, -1.0])
    u = jnp.ones(ntmpl) * jnp.inf # [jnp.inf, jnp.inf])

    qp = BoxCDQP()
    init = jnp.ones(ntmpl)
    sol = qp.run(init, params_obj=(Q, q), params_ineq=(l, u)).params

    return sol


# jax.config.update("jax_enable_x64", True)


"""
solve batched non-negative least squares (nnls) problems

min_x    |Fx - g|^2
st        x >= 0
"""

# @jit
# def form_qp(F, g):
#     # convert the least squares to qp form
#     n = F.shape[1]
#     Q = F.T @ F
#     q = -F.T @ g
#     G = -jnp.eye(n)
#     h = jnp.zeros(n)
#     A = jnp.zeros((0, n))
#     b = jnp.zeros(0)
#     return Q, q, A, b, G, h

# @jit
# def nnls(F, g):
#     Q, q, A, b, G, h = form_qp(F, g)
#     x = qpax.solve_qp_primal(Q, q, A, b, G, h)
#     return x


# from scipy.optimize import nnls as nnls2

@jax.jit
def template_lsq_(fnu_i, efnu_i, A, TEFz, zp):
    sh = A.shape

    # Valid fluxes
    ok_band = (efnu_i/zp > 0) & jnp.isfinite(fnu_i) & jnp.isfinite(efnu_i)

    # FIXME prefilter MIN_VALID_FILTERS.

    # if jnp.where(ok_band, 1, 0).sum() < MIN_VALID_FILTERS:
    #     # coeffs_i = np.zeros(sh[0])
    #     # fmodel = np.dot(coeffs_i, A)
    #     # return np.inf, np.zeros(A.shape[0]), fmodel, None
    #     return jnp.inf, jnp.zeros(sh[0], dtype="f"), jnp.zeros(sh[1], dtype="f"), None

    var = efnu_i**2 + (TEFz*jnp.maximum(fnu_i, 0.))**2
    rms = jnp.sqrt(var)

    # # Nonzero templates
    # ok_temp = (np.sum(A, axis=1) > 0)
    # if ok_temp.sum() == 0:
    #     coeffs_i = np.zeros(sh[0])
    #     fmodel = np.dot(coeffs_i, A)
    #     return np.inf, np.zeros(A.shape[0]), fmodel, None

    # Least-squares fit

    # Ax = (A/rms).T[ok_band,:]*1
    # Ax = (A/rms).T*1
    A = jnp.where(ok_band[jnp.newaxis, :], A, 0)
    Ax = (A/rms).T*1
    # Ax = jnp.where(ok_band[:, jnp.newaxis], (A/rms).T*1, 0)
    fnu_i = jnp.where(ok_band, fnu_i, 0)

    # coeffs_i, rnorm = nnls(Ax, (fnu_i/rms)[ok_band])
    coeffs_i = nnls(A.T, fnu_i)
    coeffs_i = nnls(Ax, (fnu_i/rms))
    # coeffs_i = jnp.zeros(sh[0], dtype="f")

    fmodel = jnp.dot(coeffs_i, A)
    # chi2_i = ((fnu_i-fmodel)**2/var)[ok_band].sum()
    chi2_i = jnp.where(ok_band, (fnu_i-fmodel)**2/var, 0).sum()

    # chi2_i = ((fnu_i-fmodel)**2/var).sum()

    coeffs_draw = None

    # return chi2_i, coeffs_i, fmodel, coeffs_draw
    return chi2_i, coeffs_i # , fmodel, coeffs_draw

template_lsq_map = jit(vmap(template_lsq_, in_axes=[0, 0, None, None, None]))
