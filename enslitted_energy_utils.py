import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time, os, sys
from tqdm import tqdm
from glob import glob
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
# from photutils.psf import fit_2dgaussian, GaussianPRF
# from scipy.stats import multivariate_normal
# from scipy.interpolate import make_smoothing_spline
from photutils.aperture import RectangularAperture

# have to set global variable plotpath wherever you're calling these functions
# See notebook "Enslitted energy demo" for setup


def pxl2um(pxl):
    '''SCALES detector: 18 um/pixel'''
    return pxl * 18.

def um2pxl(um):
    '''SCALES detector: 18 um/pixel'''
    return um / 18.


def find_trace_midpoints(res, rot_angle):
    '''
    Inputs:
    res       :: results array (56,8,108,108)
    rot_angle :: rotation angle of trace, CCW from horizontal axis (scalar, degrees)
    
    Returns:
    midpt_x   :: (108,108) array of x-coordinates of midpoint of traces
    midpt_y   :: (108,108) array of y-coordinates of midpoint of traces
    '''
    
    # lol doing fancy smoothing worked worse than just taking the centroid of the wavelength-centered lenslet spot

    midpt_x = res[np.shape(res)[0]//2, 0, :, :] # (103, 110)
    midpt_y = res[np.shape(res)[0]//2, 1, :, :] # (103, 110)

    return midpt_x, midpt_y



def measure_enslitted_energy(data, cenx, ceny, rot_angle, maxwidth=10.):
    '''
    Measure enslitted energy of single trace at single wavelength.
    data       :: 2048x2048 slice of datacube
    cenx, ceny :: (x,y) positions of centroid of lenslet spot at same wavelength as cube slice
    rot_angle  :: angle of dispersion (CCW from horizontal) as measured for this dataset
    N          :: number of samples of enslitted energy
    maxwidth   :: max slit width to measure spot flux [px]
    '''
    
    # Vary slit width from 0px to [maxwidth] px
    slit_widths = np.arange(0.01, maxwidth+0.05, step=0.5)
    N = len(slit_widths)
    flx = np.full(N, np.nan)
    aps = np.empty(N, object)

    # Measure flux through every slit width
    for j, sw in enumerate(slit_widths):
        
        # Place aperture (with varying slitwidth) over spot
        ap = RectangularAperture((cenx, ceny), w=15., h=sw, theta=np.deg2rad(rot_angle))
        aps[j] = ap

        # Record flux inside aperture
        flx[j] = ap.do_photometry(data)[0][0]
    
    return slit_widths, flx, aps



def measure_EE_bywav(cube, lams, res, trace_index, rot_angle, maxwidth=10., plot=True, plotpath=None):
    '''
    Measure enslitted energy of single lenslet across all wavelengths.
    '''
    
    sw_arr = np.arange(0+0.01, maxwidth+0.05, step=0.5)
    N = len(sw_arr)
    print('NUM SAMPLES:', N, 'MAXWIDTH:', maxwidth)
    
        
    # Calculate enslitted energy of trace across wavelength slices
    fluxes = np.full((np.shape(cube)[0], N), np.nan)
    apertures = np.empty((np.shape(cube)[0], N), object)

    for i in range(np.shape(cube)[0]):
        
        # Identify centroid of lenslet spot
        cenx, ceny = res[i, 0, *trace_index], res[i, 1, *trace_index]
        
        # Measure enslitted E of this trace at ith wavelength slice
        slit_widths, flx, aps = measure_enslitted_energy(cube[i], cenx, ceny, rot_angle, maxwidth=maxwidth)
        
        # Record results
        fluxes[i,:] = flx
        apertures[i,:] = aps
        
    # If requested, make figure showing results
    if plot:
        plot_EE_bywav(cube, lams, res, trace_index, rot_angle, slit_widths, fluxes, plotpath=plotpath)
    
    return slit_widths, fluxes, apertures



def plot_EE_bywav(cube, lams, res, trace_index, rot_angle, slit_widths, fluxes, plotpath=None):
    '''
    Make 3-panel plot showing enslitted energy for individual trace, measured across all 
    wavelength slices in given datacube. 
    
    rot_angle :: angle of dispersion (CCW from +X) in DEGREES
    '''
    
    # Set wavelength slice to plot as example
    ex = 15
                
    # Retrieve midpoints of all traces
    midpt_x, midpt_y = find_trace_midpoints(res, rot_angle)

    # Identify (x,y) coordinates of midpoint of this individual trace
    midx, midy = midpt_x[trace_index], midpt_y[trace_index]

    # Make big aperture for plotting
    midap = RectangularAperture((midx, midy), w=60., h=6., theta=np.deg2rad(rot_angle))

    # Make aperture for example wavelength
    cenx, ceny = res[ex, 0, *trace_index], res[ex, 1, *trace_index]
    exap6 = RectangularAperture((cenx, ceny), w=15., h=6., theta=np.deg2rad(rot_angle))
    exap10 = RectangularAperture((cenx, ceny), w=15., h=10., theta=np.deg2rad(rot_angle))
    
    # Identify indices in fluxes array where width=6px, width=10px
    loc6 = np.argmin(np.abs(slit_widths-6.))
    loc10 = np.argmin(np.abs(slit_widths-10.))

    
    
    
    # Define figure
    fig = plt.figure(figsize=(10, 8), layout="constrained")
    gs = GridSpec(2, 22, figure=fig, wspace=0.4, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, :9]) # squish imshow
    cax1 = fig.add_subplot(gs[0, 9])
    ax2 = fig.add_subplot(gs[0, 12:21]) # slice imshow
    cax2 = fig.add_subplot(gs[0, 21])
    ax3 = fig.add_subplot(gs[1, :]) # enslitted energy line plot
#         ax4 = fig.add_subplot(gs[2, :]) # enslitted energy line plot


    # Imshow squish
    squish = np.sum(cube, axis=0)
    im1 = ax1.imshow(squish, origin='lower')
    ax1.scatter(midx, midy, marker='.', color='orange', label='Trace midpoint')
    fig.colorbar(mappable=im1, cax=cax1, label='Counts', shrink=0.2)

    midap.plot(ax=ax1, color='r', label='Slit width = 6 px')

    ax1.set_xlim(midx-50,midx+50)
    ax1.set_ylim(midy-50,midy+50)
    ax1.set_title(r'Example trace: LowRes-L (2.8-4.25$\mu$m)')
    ax1.legend()


    # Imshow example cube slice
    im2 = ax2.imshow(cube[ex], origin='lower')
    fig.colorbar(mappable=im2, cax=cax2, label='Counts', shrink=0.2)

    midap.plot(ax=ax2, color='r', ls=':')
    exap6.plot(ax=ax2, color='r', label='Slit width = 6 px')
    exap10.plot(ax=ax2, color='orange', label='Slit width = 10 px')

    ax2.set_xlim(midx-50,midx+50)
    ax2.set_ylim(midy-50,midy+50)
    ax2.set_title(fr'Example slice: $\lambda$={lams[ex]:0.2f}$\mu$m')
    ax2.legend()


    # Show enslitted energy curve

    colors = plt.cm.viridis(np.linspace(0, 1, np.shape(cube)[0]))

    for i in range(np.shape(cube)[0]):
        if i%4==0: # plot subset of curves for legibility
            ax3.plot( slit_widths, fluxes[i,:]/fluxes[i,loc10], 'o-', alpha=0.5, color=colors[i] )

    secax = ax3.secondary_xaxis('top', functions=(pxl2um, um2pxl))
    secax.set_xlabel(r'Slit width ($\mu$m)')

    ax3.set_xlabel('Slit width (px)')
    ax3.set_ylabel('Flux within 6px of trace')
    ax3.grid(ls=':')
    ax3.axvline(6., ls='--', color='gray')
    
    # Annotate enslitted energy at 6x averaged across all wavelengths
    
    ee_avg = np.nanmean( fluxes[:,loc6] / fluxes[:,loc10] )
    ee_std = np.nanstd( fluxes[:,loc6] / fluxes[:,loc10] )
    
    ax3.annotate(f'Enslitted energy @ 6px (108$\mu$m): \n{100*ee_avg:0.2f}$\pm${100*ee_std:0.2f}%', 
                 xy=(6.5, 0.7), xycoords='data',
                 size=10, ha='left', va='top',
                 bbox=dict(boxstyle='round', fc='w'))


    # Custom legend for enslitted energy plot

    custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                    Line2D([0], [0], color=colors[np.shape(cube)[0]//2], lw=4),
                    Line2D([0], [0], color=colors[np.shape(cube)[0]-1], lw=4)]

    ax3.legend(custom_lines, [fr'$\lambda$ = {lams[0]:0.2f}$\mu$m', 
                              fr'$\lambda$ = {lams[np.shape(cube)[0]//2]:0.2f}$\mu$m', 
                              fr'$\lambda$ = {lams[-1]:0.2f}$\mu$m'])


    # Save figure
    if plotpath==None: plotpath = os.getcwd()
    plt.savefig(f'{plotpath}/EE_trace_{trace_index[0]}_{trace_index[1]}.png')
    
    return 0



def measure_EE_byslice(data, res, wav_index, rot_angle, maxwidth=10., plot=True):
    '''
    Measure enslitted energy of 108x108 grid of lenslets for single wavelength slice of datacube.
    '''
    
    # Figure out what shape slit_widths, fluxes will be 
    sw_arr = np.arange(0+0.01, maxwidth+0.05, step=0.5)
    N = len(sw_arr)    
        
    # Calculate enslitted energy of trace across wavelength slices
    fluxes = np.full((*np.shape(res[wav_index, 0, :]), N), np.nan)
    apertures = np.empty((*np.shape(res[wav_index, 0, :]), N), object)
    

    # Measure EE for 108x108 grid of lenslet spots
    for trace_index in np.ndindex(np.shape(res[wav_index, 0, :])):
        
        # Identify centroids of 108x108 grid of lenslet spots
        cenx, ceny = res[wav_index, 0, *trace_index], res[wav_index, 1, *trace_index]
        
        # Check if spot centroids are nan
        if np.isfinite(cenx) and np.isfinite(ceny):
        
            # Measure enslitted E of this trace at ith wavelength slice
            slit_widths, flx, aps = measure_enslitted_energy(data, cenx, ceny, rot_angle, maxwidth=maxwidth)

            # Record results
            fluxes[*trace_index, :] = flx
            apertures[*trace_index, :] = aps
            
            
    # FIXME add plotting
        
    return slit_widths, fluxes, apertures

