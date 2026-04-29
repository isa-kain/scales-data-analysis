from scipy import sparse
import os
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, shift
from photutils.centroids import centroid_sources
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import pandas as pd
from pathlib import Path
import glob
import pandas as pd

from rectmat_gen_funcs import *

'''
Steph's code to generate the L-band monochrometer scan from CD3
'''


nstart = 2570
nstop = 2880

indir = 'redux/' # location of raw-ish (quicklook) data
pref = '20251027_IFS_CDS_0'
suff = '_quicklook.fits'

ims_all = []
lams_all = []

# Extract images, monochrometer wavelength from each file in quicklook scan
for i in range(nstart,nstop+1):
    if i!=2633:
        hdu = pyfits.open(indir+pref+str(i)+suff,memmap=False)
        hdr = hdu[0].header
        im = hdu[0].data
        lam = float(hdr['MONOWAVE'])
        hdu.close()
        ims_all.append(im)
        lams_all.append(lam)
ims_all = np.array(ims_all)
lams_all = np.array(lams_all)

# Sum together images at identical monochrometer wavelengths
lams_u = np.unique(lams_all)
ims = []
for lam in lams_u:
    im = np.sum(ims_all[np.where(lams_all == lam)], axis=0)
    im[423,1995] = im[422,1994]
    im[1200,1731] = im[1198,1729]
    im[957,1866] = im[955,1864]
    im[1687,1591] = im[1685,1589]
    im[1373,2037] = im[1371,2035]
    im[1355,1534] = im[1353,1532]
    im[1921,1879] = im[1919,1877]
    im[1933,1958] = im[1931,1956]
    im[2029,2015] = im[2027,2013]
    im[176,369] = im[174,367]
    im[1564,35] = im[1562,33]
    im[567,1763] = im[565,1761]
    im[1088,187] = im[1085,184]
    im[360,1728] = im[358,1726]
    im[548,1902] = im[546,1900]
    ims.append(im)
ims = np.array(ims)

# Save 56x103x110 cube, 312x103x110 cube, len-312 wavelengths, len-56 wavelengths

pyfits.writeto('20251027_mono_Lnew_raw_allLams.fits',ims-np.median(ims,axis=0),overwrite=True)

pyfits.writeto('Lnew_allLams.fits',lams_u,overwrite=True)

ims = ims[3:59]
lams_u = lams_u[3:59]
lams_u = lams_u/1000.0
pyfits.writeto('Lnew_lams.fits',np.array(lams_u),overwrite=True)

pyfits.writeto('20251027_mono_cube_L_new.fits',ims,overwrite=True)


# idk what this is? what bitmap?
bpimcube = pyfits.getdata('20251027_mono_Lnew_bpcube.fits')
bpim2 = pyfits.getdata('cd3_bpmap.fits')

redo_calibs=False
if redo_calibs==True:
    ims_cal = []
    for ii in range(len(ims)):
        bounds = 4+510*np.array(range(5))
        bias = np.zeros([2048,2048])
        for i in range(4):
            xstart = bounds[i]
            xstop = bounds[i+1]
            arr = ims[ii,:,xstart:xstop]
            testmed = np.nanmedian(arr[np.where(arr < np.nanmedian(arr)+3*np.nanstd(arr))])
            med = np.nanmedian(arr)
            bias[:,xstart:xstop] = med
        im_cal = ims[ii]-bias

        colbias = np.zeros([2048,2048])
        for x in range(2048):
            arr = im_cal[:,x]
            med = np.nanmedian(arr)
            colbias[:,x] = med
        im_cal = im_cal-colbias


        rowbias = np.zeros([2048,2048])
        for y in range(2048):
            arr = im_cal[y]
            med = np.nanmedian(arr)
            rowbias[y] = med

        im_cal = im_cal-rowbias
        for i in range(len(bpim2)):
            for j in range(len(bpim2[i])):
                if bpim2[i,j]==1.0:
                    bpbox = bpim2[i-1:i+2,j-1:j+2]
                    if 0.0 in bpbox:
                        imbox = im_cal[i-1:i+2,j-1:j+2]
                        im_cal[i,j] = np.median(imbox[np.where(bpbox!=1.0)])
        ims_cal.append(im_cal)


    ims_cal = np.array(ims_cal)
    pyfits.writeto('20251027_mono_Lnew_bpcorrected_2.fits',ims_cal,overwrite=True)

else:
    ims_cal = pyfits.getdata('20251027_mono_Lnew_bpcorrected_2.fits')


if os.path.isfile('L_C2_rectmat_new_260227.npz')==False:
    spots = find_all_spots(ims_cal,lams_u,plot_im=False)
    spot_tracks = track_sequentially(spots, max_match_distance=6)
    spot_tracks_u = remove_spot_dups(spot_tracks,lams_u)
    avgs = find_avg_spotpos(spot_tracks_u,2.9,4.15,show_plots=True)
    avgs_new = remove_silos(avgs)
    final_posns = get_lensarr_xy(avgs_new)
    posarr = make_posarr(ims_cal,final_posns,spot_tracks_u,show_plots=True)
    pyfits.writeto('L_posarr_new_260227.fits',posarr,overwrite=True)
    C2_rmat = gen_C2_rectmat(ims_cal,posarr)
    QL_rmat = gen_QL_rectmat(ims_cal,posarr)
    sparse.save_npz('L_QL_rectmat_new_260227.npz',QL_rmat)
    sparse.save_npz('L_C2_rectmat_new_260227.npz',C2_rmat)
else:
    QL_rmat = sparse.load_npz('L_QL_rectmat_new_260227.npz')
    C2_rmat = sparse.load_npz('L_C2_rectmat_new_260227.npz')
    posarr = pyfits.getdata('L_posarr_new_260227.fits')

print(lams_u)
print(posarr.shape)
test_incube = np.zeros([len(lams_u),posarr.shape[1],posarr.shape[2]])
test_incube[0] = 1.0
testim = np.array(C2_rmat*np.matrix(test_incube.reshape([np.prod(test_incube.shape),1]))).reshape([2048,2048])

f = plt.figure()
f.add_subplot(121)
plt.imshow(testim)
plt.xlim(0,100)
plt.ylim(0,100)
plt.colorbar()
f.add_subplot(122)
plt.imshow(ims_cal[0])
plt.xlim(0,100)
plt.ylim(0,100)
plt.colorbar()
plt.show()

# THESE LINES DO CUBE (aperture photometry on trace; sums up all monochrometer images to create de-facto flat.)

test_inim = np.sum(ims_cal,axis=0)
testcube = np.array(QL_rmat*np.matrix(test_inim.reshape([2048*2048,1]))).reshape([len(lams_u),posarr.shape[1],posarr.shape[2]])
testflat = np.sum(testcube,axis=0)
plt.imshow(testflat)
plt.show()
