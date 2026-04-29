import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
# import pyfits
import time, os, sys
from glob import glob
# from matplotlib.colors import LogNorm
# from matplotlib.gridspec import GridSpec
# from photutils.psf import fit_2dgaussian, GaussianPRF
# from scipy.stats import multivariate_normal
# from photutils.detection import DAOStarFinder
# from astropy.stats import sigma_clipped_stats

sys.path.append('/Users/isabelkain/Desktop/SCALES/scales-data-analysis')
sys.path.append('/Users/isabelkain/Desktop/SCALES/scales-data-analysis/psf-fitting-tool')
from utils import *
from do_psf_fitting import *
from rectmat_gen_funcs import *

# %load_ext autoreload
# %autoreload 2

#########################
## IMPORT & CLEAN DATA ##
#########################

# Set data path
datapath = '/Users/isabelkain/Desktop/SCALES/CD4/data/IFS_img_quality'

# Retrieve K-filter, dism=mirror IFS image from CD4
fname = '20260127_IFS_UTR_00257_Kband_mirror'
imgK = fits.getdata(f'{datapath}/20260127_IFS_UTR_00257_ql.fits')
dark = fits.getdata(f'{datapath}/20260127_IFS_UTR_00262_ql.fits') / 50.

# Subtract dark from image
imgK -= dark

# Mask hot pixels (y,x)

# upper left corner
imgK[2039, 25] = 0. #imgK[26,2040] # checked, top left biggie
imgK[2043, 111] = 0. # imgK[112,2044]
imgK[1995, 46] = 0. # imgK[47,1996]

# upper right corner
imgK[1989, 1998] = imgK[1988, 1997] # checked
imgK[2043, 1991] = imgK[2042, 1990] # checked
imgK[2041, 1997] = imgK[2040, 1996] # checked
imgK[1956, 2021] = imgK[1957, 2022] # biggie, checked
imgK[2003, 2043] = imgK[2002, 2042] # checked
imgK[2022, 2033] = imgK[2021, 2032] # checked
# imgK[1669, 1916] = imgK[1668, 1915]
# imgK[1909, 2009] = imgK[1908, 2008]
# imgK[1921, 1879] = imgK[1920, 1878]
# imgK[1742, 1942] = imgK[1743, 1941]

# lower left corner
imgK[64, 86] = imgK[65, 87]
imgK[11, 78] = imgK[12, 79] # checked
imgK[129, 46] = imgK[130, 47]
imgK[66, 277] = imgK[67, 278]

# lower right edge
imgK[5, 1791] = imgK[6, 1791]
imgK[5, 1854] = imgK[6, 1854]
imgK[48, 1465] = imgK[49, 1466]

# lower right corner
imgK[34, 2014] = imgK[33, 2013] # checked
imgK[51, 2009] = imgK[50, 2010] # checked
imgK[50, 2010] = imgK[49, 2011] # checked
imgK[99, 1995] = imgK[100, 1996] # checked
imgK[31, 2040] = imgK[32, 2041] # checked
imgK[8, 2008] = imgK[9, 2009] # checked
imgK[6, 1996] = imgK[7, 1997] # checked
imgK[6, 1993] = imgK[7, 1994] # checked

# # Show hot pixel corrected image
# plt.figure()
# plt.imshow(imgK, origin='lower')
# plt.colorbar()
# plt.show()


#########################
## DO SPOT-FINDING     ##
#########################

# Assemble single frame into fake cube
fake_cube = np.array([imgK,imgK])
fake_wavl = np.array([1.95, 2.45]) # microns

# If this were a monochrometer scan, handle compressing those wavelengths

# Find spots
spots = find_all_spots(fake_cube, fake_wavl, plot_im=False)

# Track sequentially (registers by finding spot w/in certain distance between sequential images. wonky 4 bad pixels)
print('(1) track_sequentially')
spot_tracks = track_sequentially(spots, max_match_distance=6)

# Remove spot dups
print('(2) remove_spot_dups')
spot_tracks_u = remove_spot_dups(spot_tracks, fake_wavl)

# Find avg spotpos
print('(3) find_avg_spotpos')
avgs = find_avg_spotpos(spot_tracks_u, 1.95, 2.45, show_plots=True) # change to broadband wavelength extents

# Remove silos (very important before get_lensarr_xy)
print('(4) remove_silos')
avgs_new = remove_silos(avgs)

# Final positions
print('(5) get_lensarr_xy') # if this runs forever, probably there is an isolated spot somewhere
final_posns = get_lensarr_xy(avgs_new, show_plots=False) # picks one (0,0) point and builds array around it, final_posns shape (n_lens_y, n_lens_x, 2)

# Register spots
print('(6) make_posarr')
posarr = make_posarr(fake_cube, final_posns, spot_tracks_u, show_plots=False)
print('Finished.')


# Save results
if not os.path.isdir(f'{datapath}/results/{fname}'):
    os.mkdir(f'{datapath}/results/{fname}')

# Since cube is fake (o.g. data is broadband mirror IFU frame), choose only one slice
fits.writeto(f'{datapath}/results/{fname}/{fname}_posarr.fits', posarr[0,:,:,:], overwrite=True) 

# C2_rmat = gen_C2_rectmat(ims_cal,posarr)
# QL_rmat = gen_QL_rectmat(ims_cal,posarr)
# sparse.save_npz('L_QL_rectmat_new_260227.npz',QL_rmat)
# sparse.save_npz('L_C2_rectmat_new_260227.npz',C2_rmat)



#########################
## DO PSF FITTING      ##
#########################

# Do 2D Gaussian fitting of each lenslet spot. Return (6 x Nx x Ny) array: 
# [x_centroid_fit, y_centroid_fit, x_fwhm, y_fwhm, theta, quality_flags]
params = (imgK, posarr[0, :, :, :])
results = fit_spots(params)

# Because mirror data (not dispersed), can't measure dispersion direction. For now, use hardcoded value.
# RERUN LATER after measuring angle of dispersion for this bandpass.
dispersion_angle = 71.58 # degrees from horizontal (18.42˚ from vertical) THIS IS THE VALUE MEASURED FOR CD3 L-BAND

# Pull out fwhm_x, fwhm_y, and theta parameters from fitting routine
fwhm_x = results[2, :, :]
fwhm_y = results[3, :, :]
theta = wrap_angles(results[4, :, :]) # rotation angle of best-fit 2D Gauss PSF model from horizontal (deg)

# Find angle difference between dispersion direction and rotation of PSF
diff = diff_angles(theta, dispersion_angle) # degrees

# Calculate FWHM along, against direction of dispersion
fwhm_dispers = calc_ellipse_radius(fwhm_x, fwhm_y, diff)
fwhm_anti_dispers = calc_ellipse_radius(fwhm_x, fwhm_y, (diff+90.)%180.)

# Add to results array. Starts out: 56x6x103x100, becomes 56x8x103x110
new_results = np.insert(results, 5, fwhm_dispers, axis=0)
new_results = np.insert(new_results, 6, fwhm_anti_dispers, axis=0)
print(np.shape(new_results))

# Save to .fits  (reminder: cenx, ceny, fwhmx, fwhmy, theta, fwhm_disp, fwhm_adisp, quality)
hdu = fits.PrimaryHDU(data=np.array(new_results))
hdu.writeto(f'{datapath}/results/{fname}_results.fits', overwrite=True)




