import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Set data path
datapath = '/Users/isabelkain/Desktop/SCALES/CD4/data/IFS_img_quality'

# Retrieve K-filter, dism=mirror IFS image from CD4
imgK = fits.getdata(f'{datapath}/20260127_IFS_UTR_00257_ql.fits')
dark = fits.getdata(f'{datapath}/20260127_IFS_UTR_00262_ql.fits') / 50.

# Subtract dark from image
imgK -= dark




avgs = np.load('/Users/isabelkain/Desktop/SCALES/scales-data-analysis/avgs.npy')

plt.figure()
plt.imshow(imgK, origin='lower')
plt.scatter(avgs[:,0],avgs[:,1], marker='.', color='red')
plt.show()