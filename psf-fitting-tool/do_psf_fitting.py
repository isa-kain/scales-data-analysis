#!/usr/bin/env python3

'''
Author: Isabel Kain
Date: 2026-01-7
Description: 

This script does PSF fitting on SCALES monochrometer IFU data. It takes the result of coarse centroiding 
by the SCALES pipeline, and a background-subtracted and reduced datacube, and fits a 2D gaussian to each individual 
lenslet spot. 

This script returns and saves a [n_wavelength x 6 x 103 x 110] numpy array, where the first dimension 
is the number of wavelength slices in the datacube, 6 is the number parameters returned by the PSF fitting routine 
(x_centroid, y_centroid, x_FWHM, y_FWHM, rotation angle of asymmetric 2D gaussian, and a fit quality flag), the last 
two dimensions are the number of lenslets in the lenslet array. If a micropupil is not found by the pipeline, the 
corresponding values will be NaNs.
'''

import concurrent.futures
from astropy.io import fits
import numpy as np
import os, sys
import time
from photutils.psf import fit_2dgaussian, GaussianPRF
from scipy.stats import multivariate_normal
from utils import *

#########################################################################################################################

# Set filepath for SCALES post-pipeline data products
datapath = '/Users/isabelkain/Desktop/SCALES/scales-data-analysis/psf-fitting-tool/data'

# Read in a background-subtracted SCALES monochrometer datacube
cube = fits.getdata(f'{datapath}/20251027_mono_cube_L_new.fits')

# Read in rough centroids from pipeline
a = fits.getdata(f'{datapath}/L_positions_new.fits')

# Set path to save output of PSF-fitting
savepath = '/Users/isabelkain/Desktop/SCALES/scales-data-analysis/psf-fitting-tool/results'

# Set filename for output of PSF-fitting
fname = 'L_psf_fitting_results' # .fits is automatically appended


#########################################################################################################################

def eccentricity(a,b):
    '''a and b are semi-major and semi-minor axes of ellipse. Can be scalar or same-shape ndarrays.'''
    return np.sqrt(1. - (b**2. / a**2.))


def wrap_angles(theta):
    '''Theta (degrees)'''
    
    theta %= 360.
    
    return theta


def diff_angles(theta, phi):
    '''
    Find difference between two angles (degrees). Assuming counterclockwise angle from the horizontal axis.
    Inputs can be either scalar or nd arrays
    '''
    
    diff = np.abs( wrap_angles(theta) - wrap_angles(phi) )
    
    return diff


def calc_ellipse_radius(a, b, phi):
    '''a and b are semimajor and semiminor axes. Offset angle of measured radius, phi, 
    is assumed to be in DEGREES'''
    
    phi_r = np.deg2rad(phi)
    
    return a * b / np.sqrt( (b * np.cos(phi_r))**2 + (a * np.sin(phi_r))**2 )


def measure_dispersion_direction(results):
    '''
    results :: (56x6x103x110) output from do_psf_fitting.py
    Returns 103x110 map of dispersion angle, medianed across wavelength
    '''
    
    # Measure shift of each individual spot centroid between adjacent wavelength slices
    x_cen_shift = np.diff(results[:,0,:,:], axis=0)
    y_cen_shift = np.diff(results[:,1,:,:], axis=0)
    print('x_cen_shift:', np.shape(x_cen_shift), np.shape(y_cen_shift))
    
    # Measure angular centroid shift between wavelength slices (ndarray 55x103x110)
    disp_dir = np.rad2deg( np.arctan(y_cen_shift/x_cen_shift) ) # angle from horizontal axis 
    print('disp_dir:', np.shape(disp_dir))

    # Measure scalar dispersion direction
    dispersion_direction = np.nanmedian(disp_dir, axis=0)
    print('dispersion_direction:', np.shape(dispersion_direction))
    
    return dispersion_direction



# def fit_spots(data, pos):
def fit_spots(params):
    '''
    params :: 56x2 list of tuples, where each tuple is (data, pos)
    data   :: (2048, 2048) slice from SCALES monochrometer cube (background-subtracted & reduced)
    pos    :: (103, 110, 6) information about the centroid of each lenslet micropupil. The 6 values in the last 
              dimension are: [x_center, x_start, x_end, y_center, y_start, y_end]
    
    This fits a 2D gaussian to each spot in a slice of monochrometer SCALES IFU data. It returns a (6,103,110)
    array: [x_centroid_fit, y_centroid_fit, x_fwhm, y_fwhm, theta, quality_flags]
    '''
    
    # Extract cube slice, centroid positions from params
    data = params[0] # 2048x2048 slice of IFU datacube
    pos = params[1]  # 103x110x6 array with info about coarse spot centroids
    
    # Drop NaN centroids (spots that fell off the detector)
    xinf = np.isfinite(pos[:,:,0])
    yinf = np.isfinite(pos[:,:,3])
    xyinf = np.logical_and( xinf, yinf )

    # Extract (x,y) coarse centroid positions as 1D array
    xpos = np.ravel(pos[xyinf, 0])
    ypos = np.ravel(pos[xyinf, 3])

    #########################
    ## DO 2D GAUSS FITTING ##
    #########################

    # Fit circular 2D Gaussians to each centroided spot with photutils
    psfphot = fit_2dgaussian_assym(data, xypos=list(zip(xpos, ypos)), fwhm=1.5, fit_shape=9, fix_fwhm=False) 

    # Extract fitting results
    phot_tbl = psfphot.results


    #########################
    ## CHECK RESULTS       ##
    #########################

    # Check if any best-fit parameters are unreasonable
    flag_check = phot_tbl['flags'] > 0           # if any flags raised
    fwhmx_check = phot_tbl['x_fwhm_fit'] > 5.    # if FWHM_x larger than certain threshold
    fwhmy_check = phot_tbl['y_fwhm_fit'] > 5.    # if FWHM_y larger than certain threshold
    xerr_check = phot_tbl['x_err'] > 0.5         # uncertainty in centroid X position
    yerr_check = phot_tbl['y_err'] > 0.5         # uncertainty in centroid Y position
    qfit_check = np.abs(phot_tbl['qfit']) > 1.0  # quality-of-fit from residuals (zero is perfect)
    cfit_check = np.abs(phot_tbl['cfit']) > 0.1  # broadness/peakiness of PSF fit (zero is perfect)

    print('****** FIT FLAGS ******')
    print('photutils fit flags:', np.sum(flag_check))
    print('FWHM_x > 5px:       ', np.sum(fwhmx_check))
    print('FWHM_y > 5px:       ', np.sum(fwhmy_check))
    print('x_err > 0.5px:      ', np.sum(xerr_check))
    print('y_err > 0.5px:      ', np.sum(yerr_check))
    print('qfit > 1.0:         ', np.sum(qfit_check))
    print('cfit > 0.1:         ', np.sum(cfit_check))
    print('***********************')

    # Raise flag if any params are out of range (ignore qfit_check, cfit_check for now, unsure how to treat)
    quality_flag = np.any([flag_check, fwhmx_check, fwhmy_check, xerr_check, yerr_check], axis=0)


    ############################
    ## READ & RESHAPE RESULTS ##
    ############################

    ## Read out parameters from photutils output table

    # Best-fit centroid position
    x_fit_dropnan = phot_tbl['x_fit'].value
    y_fit_dropnan = phot_tbl['y_fit'].value

    # FWHM in x, y directions
    x_fwhm_dropnan = phot_tbl['x_fwhm_fit'].value
    y_fwhm_dropnan = phot_tbl['y_fwhm_fit'].value

    # Orientation angle of ellipse
    theta_dropnan = phot_tbl['theta_fit'].value


    ## Insert NaN values that were initially dropped to return all arrays to 11330-length array

    # Create temp array
    temp = np.zeros(len(np.ravel(xyinf))) # xyinf=True means value is finite
    temp[~np.ravel(xyinf)] = np.nan

    # Reconstruct len 11330 best-fit centroid positions
    x_fit_ravel = np.copy(temp)
    x_fit_ravel[np.ravel(xyinf)] = x_fit_dropnan

    y_fit_ravel = np.copy(temp)
    y_fit_ravel[np.ravel(xyinf)] = y_fit_dropnan

    # Reconstruct len 11330 FWHM in x, y directions
    x_fwhm_ravel = np.copy(temp)
    x_fwhm_ravel[np.ravel(xyinf)] = x_fwhm_dropnan

    y_fwhm_ravel = np.copy(temp)
    y_fwhm_ravel[np.ravel(xyinf)] = y_fwhm_dropnan

    # Reconstruct len 11330 orientation angle of ellipse
    theta_ravel = np.copy(temp)
    theta_ravel[np.ravel(xyinf)] = theta_dropnan

    # Reconstruct len 11330 quality flags array
    quality_ravel = np.copy(temp)
    quality_ravel[np.ravel(xyinf)] = quality_flag
    
    
    # Reshape 6 fitting parameters

    x_fit = np.reshape(x_fit_ravel, np.shape(pos[:,:,0]), order='C')
    y_fit = np.reshape(y_fit_ravel, np.shape(pos[:,:,3]), order='C')
    x_fwhm = np.reshape(x_fwhm_ravel, np.shape(pos[:,:,0]), order='C')
    y_fwhm = np.reshape(y_fwhm_ravel, np.shape(pos[:,:,3]), order='C')
    theta = np.reshape(theta_ravel, np.shape(pos[:,:,0]), order='C')
    quality = np.reshape(quality_ravel, np.shape(pos[:,:,0]), order='C')

    # Save and return results
    results = np.array([x_fit, y_fit, x_fwhm, y_fwhm, theta, quality])
    
    return results


#########################################################################################################################


if __name__ == "__main__":
    
    
    # Set up inputs as list of tuples [(datacube_slice, centroid_positions), …]
    
#     param_list = [(cube[i], a[i, :, :, :]) for i in range(0, np.shape(cube)[0])]
    param_list = [(cube[i], a[i, :, :, :]) for i in range(0, 5)]
    print(len(param_list))


    ## Execute PSF routine for each slice using multiprocessing
    
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(fit_spots, param) for param in param_list]
        results = [f.result() for f in futures] 
        print(np.shape(results))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Execution time for fitting {len(param_list)} slices: {execution_time} s.')
    
    # Convert list to array
    results = np.array(results)
    
    
    ## Calculate FWHM along direction of dispersion & add to results
    
    # Measure the direction of dispersion from centroid shifts
    dispersion_map = measure_dispersion_direction(results) # returns 103x110 map
    print('dispersion_map:', np.shape(dispersion_map))
    dispersion_angle = np.nanmean(dispersion_map[60:80, 60:80]) # sample most uniform region of image
    
    # Pull out fwhm_x, fwhm_y, and theta parameters from fitting routine
    fwhm_x = results[:, 2, :, :]
    fwhm_y = results[:, 3, :, :]
    theta = wrap_angles(results[:, 4, :, :])

    # Find angle difference between dispersion direction and rotation of PSF
    diff = diff_angles(theta, dispersion_angle)
    
    # Calculate FWHM along direction of dispersion
    fwhm_dispers = calc_ellipse_radius(fwhm_x, fwhm_y, diff)
    
  
    ## Save results
    
    # Add to results array. Starts out: 56x6x103x100, becomes 56x8x103x110
    new_results = np.insert(results, 5, fwhm_dispers, axis=1)
    new_new_results = np.insert(new_results, 6, dispersion_map, axis=1)
    print(np.shape(new_new_results))
    
    # Save to .fits
    hdu = fits.PrimaryHDU(data=np.array(new_new_results))
    hdu.writeto(f'{savepath}/{fname}.fits', overwrite=True)


