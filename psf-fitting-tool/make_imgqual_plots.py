import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time, os, sys
from glob import glob
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from utils import *
from do_psf_fitting import eccentricity


###################################
## SET THESE VALUES              ##
###################################

# TODO change this later to loop through all available results

# If analyzing a cube, this should be true; change to False if looking at a single 2048x2048 frame
iscube = False 

# Identify filenames to make plots from
# basepath = '/Users/isabelkain/Desktop/SCALES/CD3/IFS_monochrometer_scans/L-prism'
# fdata = f'{basepath}/data/20251027_mono_cube_L_new.fits'
# flams = f'{basepath}/Lnew_lams.fits'
# fres = f'{basepath}/L_psf_fitting_results.fits'

basepath = '/Users/isabelkain/Desktop/SCALES/CD4/IFS_img_quality/20260127_IFS_UTR_00257_Kband_mirror'
fdata = f'{basepath}/data/20260127_IFS_UTR_00257_ql.fits'
fres = f'{basepath}/20260127_IFS_UTR_00257_Kband_mirror_results.fits'


# Where/how should plots be saved?
plotpath = f'{basepath}/plots'
plotname = '20260127_IFS_UTR_00257_Kband_mirror' #'Lcube_CD3' # filename will be e.g. /plotpath/PLOTNAME_hist_ecc.png


###################################
## READ IN DATA, RESULTS         ##
###################################

if iscube:
    cube = fits.getdata(fdata)
    lams = fits.getdata(flams)
else:
    frame = fits.getdata(fdata)
    
results = fits.getdata(fres)

# Pull out information inside results

if iscube:

    cen_x = results[:, 0, :, :]
    cen_y = results[:, 1, :, :]
    fwhm_x = results[:, 2, :, :]
    fwhm_y = results[:, 3, :, :]
    theta = results[:, 4, :, :]
    fwhm_disp = results[:, 5, :, :]
    fwhm_adisp = results[:, 6, :, :]
    quality = results[:, 7, :, :]
    
else:

    cen_x = results[0, :, :]
    cen_y = results[1, :, :]
    fwhm_x = results[2, :, :]
    fwhm_y = results[3, :, :]
    theta = results[4, :, :]
    fwhm_disp = results[5, :, :]
    fwhm_adisp = results[6, :, :]
    quality = results[7, :, :]

# Calculate spot eccentricity
fwhm_max = np.maximum(fwhm_x, fwhm_y)
fwhm_min = np.minimum(fwhm_x, fwhm_y)
ecc = eccentricity(fwhm_max, fwhm_min)

###################################
## MAKE CUBE PLOTS               ##
###################################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~~ IMAGES ~~~~~~~~~~~~~~~~~~~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


## ~~ FWHM ~~~~~~~~~~~~~~~~~~~~~ ##

if iscube:
    fwhmd_map = np.nanmedian(fwhm_disp, axis=0)
    fwhma_map = np.nanmedian(fwhm_adisp, axis=0)
else:
    fwhmd_map = np.copy(fwhm_disp)
    fwhma_map = np.copy(fwhm_adisp)  

fig, ax = plt.subplots(1, 2, figsize=(12,5))
plt.tight_layout()  

im = ax[0].imshow(fwhmd_map, origin='lower', vmin=1.6, vmax=3.3)
ax[1].imshow(fwhma_map, origin='lower', vmin=1.6, vmax=3.3)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.10, 0.03, 0.83])
fig.colorbar(im, cax=cbar_ax, label='FWHM [px]')

ax[0].set_title(r'Lenslet spot FWHM ($\parallel$ dispersion)')
ax[1].set_title(r'Lenslet spot FWHM ($\perp$ dispersion)')

plt.savefig(f'{plotpath}/{plotname}_map_fwhm.png', bbox_inches='tight')



## ~~ ECCENTRICITY ~~~~~~~~~~~~~ ##

if iscube:
    ecc_map = np.nanmedian(ecc, axis=0)
else:
    ecc_map = np.copy(ecc)

fig, ax = plt.subplots(1, 1, figsize=(6,5))

im = ax.imshow(ecc_map, origin='lower', vmin=0)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.115, 0.05, 0.76])
fig.colorbar(im, cax=cbar_ax, label='Eccentricity')

ax.set_title('Lenslet spot eccentricity')
ax.set_xlabel('[spaxels]')
ax.set_ylabel('[spaxels]')

plt.savefig(f'{plotpath}/{plotname}_map_ecc.png', bbox_inches='tight')



## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~~ HISTOGRAMS ~~~~~~~~~~~~~~~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##



## ~~ FWHM ~~~~~~~~~~~~~~~~~~~~~ ##

if iscube:
    
    ## ~~ FWHM || ~~~~~~~~~~~~~~~~~~ ##

    linax = np.linspace(0.5, 4.0, 17)
    colors = plt.cm.viridis(np.linspace(0, 1, np.shape(fwhm_disp)[0]))
    transcolors = np.copy(colors)
    
    fig, ax = plt.subplots(1, 2, figsize=(12,4), sharex=True)

    for i in range(np.shape(fwhm_disp)[0]):
        transcolors[i][3] = 0.2

    for i in range(np.shape(fwhm_disp)[0]):
        if i%8==0:
            ax[0].hist( np.ravel(fwhm_disp[i,:,:]), bins=linax, histtype='stepfilled', color=transcolors[i], 
                       edgecolor=colors[i], lw=4, label=fr'$\lambda$ = {lams[i]:0.2f}')

    ax[1].hist( np.ravel(fwhm_disp), bins=linax, histtype='step', color='k', lw=4, 
               label=fr'$\lambda$ = {lams[0]:0.2f} - {lams[-1]:0.2f}' )

    ax[0].legend()
    ax[1].legend()
    ax[0].grid(':')
    ax[1].grid(':')
    ax[0].set_xlabel(r'Lenslet spot FWHM ($\parallel$ dispersion)')
    ax[1].set_xlabel(r'Lenslet spot FWHM ($\parallel$ dispersion)')
    
    plt.savefig(f'{plotpath}/{plotname}_hist_fwhm||.png', bbox_inches='tight')
    
    
    ## ~~ FWHM |_ ~~~~~~~~~~~~~~~~~~ ##

    fig, ax = plt.subplots(1, 2, figsize=(12,4), sharex=True)

    for i in range(np.shape(fwhm_adisp)[0]):
        if i%8==0:
            ax[0].hist( np.ravel(fwhm_adisp[i,:,:]), bins=linax, histtype='stepfilled', color=transcolors[i], 
                       edgecolor=colors[i], lw=4, label=fr'$\lambda$ = {lams[i]:0.2f}')

    ax[1].hist( np.ravel(fwhm_adisp), bins=linax, histtype='step', color='k', lw=4, 
               label=fr'$\lambda$ = {lams[0]:0.2f} - {lams[-1]:0.2f}' )

    ax[0].legend()
    ax[1].legend()
    ax[0].grid(':')
    ax[1].grid(':')
    ax[0].set_xlabel(r'Lenslet spot FWHM ($\perp$ dispersion)')
    ax[1].set_xlabel(r'Lenslet spot FWHM ($\perp$ dispersion)')

    plt.savefig(f'{plotpath}/{plotname}_hist_fwhm|_.png', bbox_inches='tight')

    
else:
    
    linax = np.linspace(0.5, 4.0, 17)

    plt.figure(figsize=(5.5,4))
    plt.hist( np.ravel(fwhm_disp), bins=linax, histtype='step', color='C0', lw=4, label=r'FWHM ($\parallel$ dispersion)' )
    plt.hist( np.ravel(fwhm_adisp), bins=linax, histtype='step', color='C1', lw=4, label=r'FWHM ($\perp$ dispersion)' )
    plt.grid(ls=':')
    plt.xlabel(r'Lenslet spot FWHM')
    plt.legend()
    
    plt.savefig(f'{plotpath}/{plotname}_hist_fwhm.png', bbox_inches='tight')


## ~~ ECCENTRICITY ~~~~~~~~~~~~~ ##

if iscube:

    fig, ax = plt.subplots(1, 2, figsize=(12,4), sharex=True)

    linax = np.linspace(0, 1, 13)
    colors = plt.cm.viridis(np.linspace(0, 1, np.shape(ecc)[0]))
    transcolors = np.copy(colors)

    for i in range(np.shape(ecc)[0]):
        transcolors[i][3] = 0.2

    for i in range(np.shape(ecc)[0]):
        if i%8==0:
            ax[0].hist( np.ravel(ecc[i,:,:]), bins=linax, histtype='stepfilled', color=transcolors[i], 
                       edgecolor=colors[i], lw=4, label=fr'$\lambda$ = {lams[i]:0.2f}')

    ax[1].hist( np.ravel(ecc), bins=linax, histtype='step', color='k', lw=4, 
               label=fr'$\lambda$ = {lams[0]:0.2f} - {lams[-1]:0.2f}' )

    ax[0].legend()
    ax[1].legend()
    ax[0].grid(':')
    ax[1].grid(':')
    ax[0].set_xlabel('Lenslet spot eccentricity')
    ax[1].set_xlabel('Lenslet spot eccentricity')
    
else:
    
    linax = np.linspace(0, 1, 13)

    plt.figure(figsize=(5.5, 4))
    plt.hist( np.ravel(ecc), bins=linax, histtype='step', color='C0', lw=4 )
    plt.grid(ls=':')
    plt.xlabel('Lenslet spot eccentricity')

plt.savefig(f'{plotpath}/{plotname}_hist_ecc.png', bbox_inches='tight')



###################################
## MAKE SINGLE-FRAME PLOTS       ##
###################################

