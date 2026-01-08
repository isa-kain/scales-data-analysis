#!/usr/bin/env python3

'''
Author: Isabel Kain
Date: 2025-12-20
Description: Utility functions for PSF fitting lenslet spots. Slightly modified from photutils fit_2d_gaussian()
'''


import warnings
from copy import deepcopy

import numpy as np
from astropy.table import QTable
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning

from photutils.centroids import centroid_com
from photutils.psf.functional_models import CircularGaussianPRF, GaussianPRF
from photutils.utils import CutoutImage
from photutils.utils._parameters import as_pair

__all__ = ['fit_2dgaussian_assym', '_make_mask']

# need import: _make_mask, 

def _make_mask(image, mask):
    def warn_nonfinite():
        warnings.warn('Input data contains unmasked non-finite values '
                      '(NaN or inf), which were automatically ignored.',
                      AstropyUserWarning)

    # if NaNs are in the data, no actual fitting takes place
    # https://github.com/astropy/astropy/pull/12811
    finite_mask = ~np.isfinite(image)

    if mask is not None:
        finite_mask |= mask
        if np.any(finite_mask & ~mask):
            warn_nonfinite()
    else:
        mask = finite_mask
        if np.any(finite_mask):
            warn_nonfinite()
        else:
            mask = None

    return mask


def fit_2dgaussian_assym(data, *, xypos=None, fwhm=None, fix_fwhm=True,
                         fit_shape=None, mask=None, error=None):
    """
    Fit a 2D Gaussian model to one or more sources in an image.

    This convenience function uses a
    `~photutils.psf.GaussianPRF` model to fit the sources using
    the `~photutils.psf.PSFPhotometry` class.

    Non-finite values (e.g., NaN or inf) in the ``data`` array are
    automatically masked.

    Parameters
    ----------
    data : 2D array
        The 2D array of the image. The input array must be background
        subtracted.

    xypos : array-like, optional
        The initial (x, y) pixel coordinates of the sources. If `None`,
        then one source will be fit with an initial position using the
        center-of-mass centroid of the ``data`` array.

    fwhm : float, optional
        The initial guess for the FWHM of the Gaussian PSF model. If
        `None`, then the initial guess is half the mean of the x and y
        sizes of the ``fit_shape`` values.

    fix_fwhm : bool, optional
        Whether to fix the FWHM of the Gaussian PSF model during the
        fitting process.

    fit_shape : int or tuple of two ints, optional
        The shape of the fitting region. If a scalar, then it is assumed
        to be a square. If `None`, then the shape of the input ``data``
        will be used.

    mask : array-like (bool), optional
        A boolean mask with the same shape as the input ``data``, where
        a `True` value indicates the corresponding element of ``data``
        is masked.

    error : 2D array, optional
        The pixel-wise Gaussian 1-sigma errors of the input
        ``data``. ``error`` is assumed to include *all* sources
        of error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) . ``error`` must have the
        same shape as the input ``data``. If a `~astropy.units.Quantity`
        array, then ``data`` must also be a `~astropy.units.Quantity`
        array with the same units.

    Returns
    -------
    result : `~photutils.psf.PSFPhotometry`
        The PSF-fitting photometry results.

    See Also
    --------
    fit_fwhm : Fit the FWHM of one or more sources in an image.

    Notes
    -----
    The source(s) are fit with a `~photutils.psf.CircularGaussianPRF`
    model using the `~photutils.psf.PSFPhotometry` class. The initial
    guess for the flux is the sum of the pixel values within the fitting
    region. If ``fwhm`` is `None`, then the initial guess for the FWHM
    is half the mean of the x and y sizes of the ``fit_shape`` values.

    Examples
    --------
    Fit a 2D Gaussian model to a image containing only one source (e.g.,
    a cutout image):

    >>> import numpy as np
    >>> from photutils.psf import CircularGaussianPRF, fit_2dgaussian
    >>> yy, xx = np.mgrid[:51, :51]
    >>> model = CircularGaussianPRF(x_0=22.17, y_0=28.87, fwhm=3.123, flux=9.7)
    >>> data = model(xx, yy)
    >>> fit = fit_2dgaussian(data, fix_fwhm=False)
    >>> phot_tbl = fit.results  # doctest: +FLOAT_CMP
    >>> cols = ['x_fit', 'y_fit', 'fwhm_fit', 'flux_fit']
    >>> for col in cols:
    ...     phot_tbl[col].info.format = '.4f'  # optional format
    >>> print(phot_tbl[['id'] + cols])
     id  x_fit   y_fit  fwhm_fit flux_fit
    --- ------- ------- -------- --------
      1 22.1700 28.8700   3.1230   9.7000

    Fit a 2D Gaussian model to multiple sources in an image:

    >>> import numpy as np
    >>> from photutils.detection import DAOStarFinder
    >>> from photutils.psf import (CircularGaussianPRF, fit_2dgaussian,
    ...                            make_psf_model_image)
    >>> model = CircularGaussianPRF()
    >>> data, sources = make_psf_model_image((100, 100), model, 5,
    ...                                      min_separation=25,
    ...                                      model_shape=(15, 15),
    ...                                      flux=(100, 200), fwhm=[3, 8])
    >>> finder = DAOStarFinder(0.1, 5)
    >>> finder_tbl = finder(data)
    >>> xypos = list(zip(sources['x_0'], sources['y_0']))
    >>> psfphot = fit_2dgaussian(data, xypos=xypos, fit_shape=7,
    ...                          fix_fwhm=False)
    >>> phot_tbl = psfphot.results
    >>> len(phot_tbl)
    5

    Here we show only a few columns of the photometry table:

    >>> cols = ['x_fit', 'y_fit', 'fwhm_fit', 'flux_fit']
    >>> for col in cols:
    ...     phot_tbl[col].info.format = '.4f'  # optional format
    >>> print(phot_tbl[['id'] + cols])
     id  x_fit   y_fit  fwhm_fit flux_fit
    --- ------- ------- -------- --------
      1 61.7787 74.6905   5.6947 147.9988
      2 30.2017 27.5858   5.2138 123.2373
      3 10.5237 82.3776   7.6551 180.1881
      4  8.4214 12.0369   3.2026 192.3530
      5 76.9412 35.9061   6.6600 126.6130
    """
    # prevent circular import
    from photutils.psf.photometry import PSFPhotometry

    # mask non-finite values
    mask = _make_mask(data, mask) # CUSTOM

    if xypos is None:
        xypos = centroid_com(data, mask=mask)
    xypos = np.atleast_2d(xypos)

    if fit_shape is None:
        fit_shape = data.shape
    else:
        fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(1, 0),
                            check_odd=True)

    flux_init = []
    for yxpos in xypos[:, ::-1]:
        cutout = CutoutImage(data, yxpos, tuple(fit_shape))
        cutout = cutout.data[np.isfinite(cutout.data)]
        flux_init.append(np.nansum(cutout))

    if isinstance(data, Quantity):
        flux_init <<= data.unit

    init_params = QTable()
    init_params['x'] = xypos[:, 0]
    init_params['y'] = xypos[:, 1]
    init_params['flux'] = flux_init

    if fwhm is None:
        fwhm = np.mean(fit_shape) / 2.0
    init_params['fwhm'] = fwhm

    model = GaussianPRF(x_fwhm=fwhm, y_fwhm=fwhm) # CUSTOM  
    model.x_fwhm.min = 0.0
    model.y_fwhm.min = 0.0
    
    model.theta.fixed = False
    
    if not fix_fwhm:
        model.x_fwhm.fixed = False
        model.y_fwhm.fixed = False
                
    phot = PSFPhotometry(model, fit_shape, progress_bar=True)
    _ = phot(data, mask=mask, error=error, init_params=init_params)

    return phot