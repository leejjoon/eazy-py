import pickle

import os
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=20' # Use 8 CPU devices

import numpy as np
from eazy import utils
# from eazy.photoz import PhotoZ as PhotoZ_, fit_by_redshift # , MULTIPROCESSING_TIMEOUT
from eazy.photoz import (MIN_VALID_FILTERS,
                         MIN_VALID_FILTERS,
                         BOUNDED_DEFAULTS)

from template_lsq import template_lsq_, template_lsq_map
from eazy.photoz import template_lsq

def fit_by_redshift(iz, z, A, fnu_corr, efnu_corr, TEFz, zp, verbose, fitter,
                    do_vmap=False):
    """
    Fit all objects in the catalog at a given reshift for parallelization
    
    Parameters
    ----------
    iz : int
        Index of the redshift grid
    
    z : float
        Redshift value
    
    A : array (NTEMP, NFILT)
        `~eazy.photoz.TemplateGrid` photometry evaluated at redshift `z`.
    
    fnu_corr, efnu_corr : array (NOBJ, NFILT)
        Flux densities and uncertainties *without* MW extinction and *with*
        `~eazy.photoz.PhotoZ.zp` zeropoint corrections
    
    TEFz : array (NFILT)
        `~eazy.templates.TemplateError` evaluated at redshift `z`.
    
    zp : array (NFILT)
        Zeropoint corrections needed to back out of `efnu_corr`
        
    verbose : int
        Prints status message if ``verbose > 2``.
    
    fitter : str
        Least-squares method for template fits.  See
        `~eazy.photoz.template_lsq`.    
        
    Returns
    -------
    iz : int
        Same as input, used for collecting results from parallel threads
    
    chi2 : array (NOBJ)
        :math:`\chi^2` of the template fits
    
    coeffs : array (NOBJ, NTEMP)
        Template normalization coefficients

    """
    NOBJ, NFILT = fnu_corr.shape#[0]
    NTEMP = A.shape[0]
    chi2 = np.zeros(NOBJ, dtype=fnu_corr.dtype)
    coeffs = np.zeros((NOBJ, NTEMP), dtype=fnu_corr.dtype)
    #TEFz = TEF(z)

    if verbose > 2:
        print('z={0:7.3f}'.format(z))

    if do_vmap:
        # msk_drop = np.sum(efnu_corr > 0, axis=1)
        iobj = np.sum((efnu_corr > 0) & np.isfinite(fnu_corr) & np.isfinite(efnu_corr),
                          axis=1) > 2

        fnu_i = fnu_corr[iobj, :]
        efnu_i = efnu_corr[iobj,:]

        _res = template_lsq_map(fnu_corr[iobj], efnu_corr[iobj], A, TEFz, zp)

        chi2[iobj], coeffs[iobj] = _res

    else:
        # for iobj in range(NOBJ):
        for iobj in range(NOBJ):

            fnu_i = fnu_corr[iobj, :]
            efnu_i = efnu_corr[iobj,:]
            ok_band = (efnu_i > 0)

            # FIXME similar filtering is doen within `template_lsq`. Not sure this is necessary.
            if ok_band.sum() < 2:
                continue

            # _res2 = template_lsq_(fnu_i[ok_band], efnu_i[ok_band], A[:, ok_band], TEFz[ok_band], zp[ok_band])
            _res = template_lsq_(fnu_i, efnu_i, A, TEFz, zp)
            # chi2[iobj], coeffs[iobj], fmodel, draws = _res
            chi2[iobj], coeffs[iobj] = _res

    return iz, chi2, coeffs

# from scipy.optimize import nnls
# import scipy.optimize

# def template_lsq(fnu_i, efnu_i, A, TEFz, zp, ndraws, fitter):
#     """
#     This is the main least-squares function for fitting templates to
#     photometry at a given redshift

#     Parameters
#     ----------
#     fnu_i : array (NFILT)
#         Flux densities, **including extinction and zeropoint corrections**

#     efnu_i : array (NFILT)
#         Uncertainties, **including extinction and zeropoint corrections**

#     A : array (NTEMP, NFILT)
#         Design matrix of templates integrated through filter bandpasses at
#         a particular redshift, z (not specified but implicit)

#     TEFz : array (NFILT)
#         `~eazy.templates.TemplateError` evaluated at same redshift as `A`.

#     zp : array (NFILT)
#         Multiplicative zeropoint corrections needed to back out from `efnu_i`
#         and test for valid data

#     ndraws : int
#         If > 0, take `ndraws` random coefficient draws from fit covariance
#         matrix

#     fitter : str
#         Template fitting method. The only stable option so far is 'nnls' for
#         non-negative least squares with `scipy.optimize.nnls`, other options
#         under development (e.g, 'bounded', 'regularized').

#     Returns
#     -------
#     chi2_i : float
#         Chi-squared of the fit

#     coeffs : array (NTEMP)
#         Template coefficients

#     fmodel : array (NFILT)
#         Flux densities of the best-fit model

#     coeffs_draw : array (`ndraws`, NTEMP)
#         Random draws from covariance matrix, if `ndraws` > 0

#     """

#     sh = A.shape


#     _ = template_lsq_(fnu_i, efnu_i, A[ok_temp], TEFz, zp, ndraws)
#     chi2_i, coeffs_i_, fmodel, coeffs_draw = _
#     coeffs_i = np.zeros(sh[0])
#     coeffs_i[ok_temp] = coeffs_i_

#     return chi2_i, coeffs_i, fmodel, coeffs_draw


# if False:
#     np.zeros(A.shape[0])

#     coeffs_i = np.zeros(sh[0])
#     coeffs_i[ok_temp] = coeffs_x




#     # Valid fluxes
#     ok_band = (efnu_i/zp > 0) & np.isfinite(fnu_i) & np.isfinite(efnu_i)
#     if ok_band.sum() < MIN_VALID_FILTERS:
#         coeffs_i = np.zeros(sh[0])
#         fmodel = np.dot(coeffs_i, A)
#         # return np.inf, np.zeros(A.shape[0]), fmodel, None

#     var = efnu_i**2 + (TEFz*np.maximum(fnu_i, 0.))**2
#     rms = np.sqrt(var)

#     # Nonzero templates
#     ok_temp = (np.sum(A, axis=1) > 0)
#     if ok_temp.sum() == 0:
#         coeffs_i = np.zeros(sh[0])
#         fmodel = np.dot(coeffs_i, A)
#         # return np.inf, np.zeros(A.shape[0]), fmodel, None

#     # Least-squares fit
#     Ax = (A/rms).T[ok_band,:]*1

#     coeffs_x, rnorm = nnls(Ax[:,ok_temp], (fnu_i/rms)[ok_band])
#     coeffs_i = np.zeros(sh[0])
#     coeffs_i[ok_temp] = coeffs_x

#     fmodel = np.dot(coeffs_i, A)
#     chi2_i = ((fnu_i-fmodel)**2/var)[ok_band].sum()

#     coeffs_draw = None

#     # return chi2_i, coeffs_i, fmodel, coeffs_draw



if True:
    k = pickle.load(open("fit_by_redshift_params.pickle", "rb"))
    iz, z, A, fnu_corr1, efnu_corr1, TEFz, zp, verbose, fitter = k
    A, fnu_corr1, efnu_corr1, TEFz, zp = [_.astype("float32")
                                          for _ in [A, fnu_corr1, efnu_corr1, TEFz, zp]]

    fnu_corr = np.vstack([fnu_corr1]*10)
    efnu_corr = np.vstack([efnu_corr1]*10)
    
    # Nonzero templates
    ok_temp = (np.sum(A, axis=1) > 0)
    if ok_temp.sum() == 0:
        # coeffs_i = np.zeros(sh[0])
        # fmodel = np.dot(coeffs_i, A)
        # return np.inf, np.zeros(A.shape[0]), fmodel, None
        chi2 = np.empty((fnu_corr.shape[0], ), dtype="float32")
        chi2.fill(np.nan)

        coeffs = np.zeros((fnu_corr.shape[0], A.shape[0]), dtype="float32")

        # return np.inf, np.zeros(A.shape[0]), np.zeros(A.shape[1]), None
        # return iz, chi2, coeffs
    else:
        _ = fit_by_redshift(iz, z, A[ok_temp], fnu_corr, efnu_corr, TEFz, zp, verbose, fitter,
                            do_vmap=True)
        iz, chi2, _coeffs = _
        coeffs = np.zeros((fnu_corr.shape[0], A.shape[0]), dtype="float32")
        coeffs[:, ok_temp] = _coeffs

    if False:
        iz_, chi2_, coeffs_ = pickle.load(open("fit_by_redhshift_results.pickle", "rb"))

    # model = A[9, :]

import jax

gpu_device = jax.devices('gpu')[0]
cpu_device = jax.devices('cpu')[0]

def test():
    with jax.default_device(cpu_device):
        _ = fit_by_redshift(iz, z, A[ok_temp], fnu_corr, efnu_corr, TEFz, zp, verbose, fitter,
                            do_vmap=False)

