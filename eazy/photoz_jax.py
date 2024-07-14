import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import BoxCDQP

# from .photoz import PhotoZ

# class PhotoZJax(PhotoZ):
#     pass


@jax.jit
def nnls(F, g):
    # jax version of nnls usin jaxopt

    ntmpl = F.shape[1]

    Q = F.T @ F
    q = -F.T @ g

    l = jnp.zeros(ntmpl)
    u = jnp.ones(ntmpl) * jnp.inf

    qp = BoxCDQP()
    init = jnp.ones(ntmpl)
    sol = qp.run(init, params_obj=(Q, q), params_ineq=(l, u)).params

    return sol


@jax.jit
def template_lsq(fnu_i, efnu_i, A, TEFz, zp):
    sh = A.shape

    # Valid fluxes
    ok_band = (efnu_i/zp > 0) & jnp.isfinite(fnu_i) & jnp.isfinite(efnu_i)

    var = efnu_i**2 + (TEFz*jnp.maximum(fnu_i, 0.))**2
    rms = jnp.sqrt(var)

    # Least-squares fit

    # for invalid band, replace the model and flux to 0. This is jax-specific.
    A = jnp.where(ok_band[jnp.newaxis, :], A, 0)
    Ax = (A/rms).T*1
    fnu_i = jnp.where(ok_band, fnu_i, 0)

    coeffs_i = nnls(Ax, (fnu_i/rms))

    fmodel = jnp.dot(coeffs_i, A)
    chi2_i = jnp.where(ok_band, (fnu_i-fmodel)**2/var, 0).sum()

    # coeffs_draw = None

    return chi2_i, coeffs_i # , fmodel, coeffs_draw

template_lsq_map = jax.jit(jax.vmap(template_lsq, in_axes=[0, 0, None, None, None]))

def main():
    import pickle
    from scipy.optimize import nnls as nnls2

    k = pickle.load(open("fit_by_redshift_params.pickle", "rb"))
    iz, z, A0, fnu_corr, efnu_corr, TEFz, zp, verbose, fitter = k
    iobj = 0
    fnu_i_ = fnu_corr[iobj]
    ok_band = fnu_i_ > 0
    sol0 = np.zeros_like(fnu_i_)

    A = A0[:, ok_band]
    fnu_i = fnu_i_[ok_band]

    sol2, _ = nnls2(A.T, fnu_i)
    F = jnp.array(A.T) # 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    g = jnp.array(fnu_i) # zeros(nband)#array([1.0, -1.0])

    sol1 = nnls(F, g)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, num=1, clear=True)
    ax.plot(fnu_i, color="0.8", lw=4)
    ax.plot(A.T @ sol1)
    ax.plot(A.T @ sol2)

def test():
    import pickle
    from scipy.optimize import nnls as nnls2

    k = pickle.load(open("fit_by_redshift_params.pickle", "rb"))
    iz, z, A, fnu_corr, efnu_corr, TEFz, zp, verbose, fitter = k

    NOBJ, NFILT = fnu_corr.shape#[0]
    NTEMP = A.shape[0]
    chi2 = np.zeros(NOBJ, dtype=fnu_corr.dtype)
    coeffs = np.zeros((NOBJ, NTEMP), dtype=fnu_corr.dtype)

    if True:
        iobj0_ = np.sum((efnu_corr > 0) & np.isfinite(fnu_corr) & np.isfinite(efnu_corr),
                       axis=1) > 2

        iobj0 = np.arange(len(iobj0_))[iobj0_] # mask to indices

        istep = 128
        # from .photoz_jax import template_lsq_map as template_lsq_jax_map
        # from .photoz_jax import template_lsq as template_lsq_jax
        template_lsq_jax_map = template_lsq_map
        template_lsq_jax = template_lsq

        i = -1  # incase len(iobj0) < istep where i won't be set.
        for i in range(0, len(iobj0) - istep, istep):
            # print(i*istep, (i+1)*istep)
            iobj = iobj0[i:i+istep]
            # print(len(iobj))
            fnu_i = fnu_corr[iobj, :]
            efnu_i = efnu_corr[iobj,:]

            _res = template_lsq_jax_map(fnu_corr[iobj], efnu_corr[iobj], A, TEFz, zp)

            chi2[iobj], coeffs[iobj] = _res

        iobj = iobj0[i:i+istep]
        if iobj:
            for iobj in range(NOBJ):
                fnu_i = fnu_corr[iobj, :]
                efnu_i = efnu_corr[iobj,:]
                ok_band = (efnu_i > 0)

                if ok_band.sum() < 2:
                    continue

                _res = template_lsq_jax(fnu_i, efnu_i, A, TEFz, zp)
                chi2[iobj], coeffs[iobj] = _res

