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

def track_sequentially(spots,max_match_distance=3):
    """
    Track spots sequentially forward through images.

    Each spot in image 0 gets a spot_id. As we move to image 1, we:
    1. Match spots from image 0 to image 1
    2. Assign same spot_id to matched spots
    3. Give new spot_ids to unmatched spots (newly appearing)

    This allows tracking diagonal motion and handles appearing/disappearing spots.

    Returns:
    --------
    tracking_df : DataFrame
        DataFrame with tracking results
    """
    n_images = len(spots)


    print(f"\nSequential tracking through {n_images} images...")

    # Initialize with first image
    coords_0 = spots[0]['coordinates']
    intensities_0 = spots[0]['intensities']
    lams_0 = spots[0]['lam']
    n_spots_0 = len(coords_0)

    # Each spot gets a unique ID
    current_spot_ids = np.arange(n_spots_0)
    next_spot_id = n_spots_0  # Counter for new spots that appear later

    # Initialize tracking data structure
    # spot_tracks[spot_id] = {image_idx: (x, y, intensity), ...}
    spot_tracks = {}
    for spot_id in range(n_spots_0):
        spot_tracks[spot_id] = {
            0: (coords_0[spot_id, 0], coords_0[spot_id, 1], intensities_0[spot_id], lams_0[spot_id])
        }

    print(f"  Image 0: {n_spots_0} spots initialized")

    # Track forward through remaining images
    prev_coords = coords_0
    prev_spot_ids = current_spot_ids

    for img_idx in range(1, n_images):
        curr_coords = spots[img_idx]['coordinates']
        curr_intensities = spots[img_idx]['intensities']
        curr_lams = spots[img_idx]['lam']
        n_curr_spots = len(curr_coords)

        if len(prev_coords) == 0:
            # No spots in previous image, all current spots are new
            new_spot_ids = np.arange(next_spot_id, next_spot_id + n_curr_spots)
            for i, spot_id in enumerate(new_spot_ids):
                spot_tracks[spot_id] = {
                    img_idx: (curr_coords[i, 0], curr_coords[i, 1], curr_intensities[i], curr_lams[i])
                }
            next_spot_id += n_curr_spots
            prev_coords = curr_coords
            prev_spot_ids = new_spot_ids
            print(f"  Image {img_idx}: {n_curr_spots} new spots (no previous spots to match)")
            continue

        # Build KDTree for current image spots
        tree = KDTree(curr_coords)

        # Find nearest spot in current image for each spot in previous image
        distances, indices = tree.query(prev_coords)

        # Track which current spots have been matched
        matched_curr_indices = set()
        curr_spot_ids = np.full(n_curr_spots, -1, dtype=int)

        n_matched = 0
        n_lost = 0

        # Match spots from previous image
        for prev_idx, (dist, curr_idx) in enumerate(zip(distances, indices)):
            if dist < max_match_distance and curr_idx not in matched_curr_indices:
                # Good match - carry forward the spot_id
                spot_id = prev_spot_ids[prev_idx]
                curr_spot_ids[curr_idx] = spot_id
                matched_curr_indices.add(curr_idx)

                # Add to trajectory
                spot_tracks[spot_id][img_idx] = (
                    curr_coords[curr_idx, 0],
                    curr_coords[curr_idx, 1],
                    curr_intensities[curr_idx],
                    curr_lams[curr_idx]
                )
                n_matched += 1
            else:
                # Spot lost (disappeared or moved too far)
                n_lost += 1

        # Handle new spots (unmatched in current image)
        n_new = 0
        for curr_idx in range(n_curr_spots):
            if curr_spot_ids[curr_idx] == -1:
                # New spot appearing
                new_spot_id = next_spot_id
                curr_spot_ids[curr_idx] = new_spot_id
                spot_tracks[new_spot_id] = {
                    img_idx: (curr_coords[curr_idx, 0], curr_coords[curr_idx, 1], curr_intensities[curr_idx], curr_lams[curr_idx])
                }
                next_spot_id += 1
                n_new += 1

        print(f"  Image {img_idx}: {n_matched} matched, {n_lost} lost, {n_new} new (total: {n_curr_spots})")

        # Update for next iteration
        prev_coords = curr_coords
        prev_spot_ids = curr_spot_ids

    total_spots = len(spot_tracks)
    print(f"\nTotal unique spots tracked: {total_spots}")

    return spot_tracks


def remove_spot_dups(spot_tracks,lams_u,maxdist=13,lmin=2.9,lmax=4.15,chx=1,chy=1):
    """
    Function to ingest multi-wavelength spot tracks, which do not have
    indices that correspond to specific lenslets,
    and find spots that fall
    on common traces. Spots on the same trace are consolidated.
    """


    #define dictionary for unique traces
    spots_u = {}
    #define dictionary for spot tracks that may
    #duplicate traces
    spots_d = {}

    uc = 0
    dc = 0
    for i in range(len(spot_tracks)):
        #if a spot track has locations for every wavelength
        #it corresponds to one lenslet and does not need to
        #be consolidated
        if len(spot_tracks[i]) == len(lams_u):
            spots_u[uc] = spot_tracks[i]
            uc += 1
        #if not, add this set of spot positions to the list
        #that may duplicate traces
        else:
            spots_d[dc] = spot_tracks[i]
            dc += 1

    #start looking for lists of spot positions that can
    #be consolidated (because they fall on one trace)

    #define list of spot track indices that are duplicates
    rem = []
    for i in range(len(spots_d)):
        keys = list(spots_d[i].keys())
        #grab spot's first x,y position and first wavelength
        x0,y0 = spots_d[i][keys[0]][0],spots_d[i][keys[0]][1]
        lam0 = spots_d[i][keys[0]][3]
        #get average wavelength and calculate expected x,y position
        #at that wavelength
        lam = 0.5*(lmin+lmax)
        xi,yi = get_trace_pos(lam,x0,y0,lam0,chx,chy,lmin,lmax)

        #if my track isn't already flagged as a duplicate
        if i not in rem:
            merged = spots_d[i]
            #loop through all other tracks
            for j in range(len(spots_d)):
                if i != j:
                    #if the other track isn't a duplicate
                    if j not in rem:
                        #get the other track's position at the central wavelegnth
                        keys = list(spots_d[j].keys())
                        x0,y0 = spots_d[j][keys[0]][0],spots_d[j][keys[0]][1]
                        lam0 = spots_d[j][keys[0]][3]
                        lam = 0.5*(lmin+lmax)
                        xj,yj = get_trace_pos(lam,x0,y0,lam0,chx,chy,lmin,lmax)
                        #get the distance between the other spot's central wavelength
                        #position and the original one
                        dist = np.sqrt((xi-xj)**2+(yi-yj)**2)
                        #if the distance between the two central wavelength positions
                        #is smaller than the maximum allowed, then they're the same
                        if dist < maxdist:
                            #append j index to list to be removed
                            rem.append(j)
                            merged = merged | spots_d[j]
            #if you accidentally merged tracks and ended up with more positions
            #than the number of wavelengths, then throw an error
            if len(merged) > len(lams_u):
                print('uh oh bad merging!!!')
                stop
            #append merged track to dictionary containing unique lenslet tracks
            spots_u[uc] = merged
            uc += 1
            #append i in dex to list to be removed
            rem.append(i)
    return spots_u



def get_trace_pos(lam,x0,y0,lam0,chx,chy,lmin,lmax,length=54,tilt=18):
    """
    Function to get the expected position of a certain wavelength
    within a trace.

    Args:
        lam: wavelength at which to calculate trace position
        x0: reference x position of trace
        y0: reference y position of trace
        lam0: reference wavelength for trace position (x0,y0)
        chx: direction of trace x movement with +ve lambda
        chy: direction of trace y movement with +ve lambda
        length: trace length in pixels (default = 54)
        tilt: trace tilt relative to vertical in deg (default = 18)

    Returns:
        xpos: trace x position at wavelength lam
        ypos: trace y position at wavelength lam
    """
    dlam = lam-lam0
    xoff = dlam/(lmax-lmin)*length*np.sin(np.radians(tilt))*chx
    yoff = dlam/(lmax-lmin)*length*np.cos(np.radians(tilt))*chy
    xpos = x0+xoff
    ypos = y0+yoff
    return xpos, ypos


def remove_silos(avgs):
    """
    Function to remove lenslet tracks that have no neighbors.

    Args:
        avgs: list of (x,y) trace positions for the average
              wavelength in the mode

    Returns:
        avgs_new: list of (x,y) trace positions where lenslets
                  that have no neighbors have been removed
    """

    #create scipy KDTree using input set of positions
    kd1 = KDTree(avgs)
    todel = []
    #loop through all lenslet spot positions
    for i in range(len(avgs)):
        #create KDTree for single lenslet position to search
        #for neighbors
        kd0 = KDTree(avgs[i:i+1])
        #query for neighbors within 33 pixels
        neighbors = kd0.query_ball_tree(kd1, 33, p=2.0, eps=0)
        #if only one spot in the large KDTree is within 33 pixels
        #of the spot in question, the spot in question has no
        #neighbors (i.e. the spot in question itself is the only
        #one found
        diff = kd1.data[neighbors] - kd0.data
        if len(diff[0])==1:
            todel.append(i)
    #delete neighborless spots from the list
    avgs_new = np.delete(avgs,todel,axis=0)
    return avgs_new

def get_lensarr_xy(avgs,show_plots=False):
    """
    Function to take clean array of spot positions and
    register them into a x,y grid of lenslets.

    Args:
        avgs: array of pixel (x,y) positions for each spot
              track at the average wavelength in the mode

    Returns:
        final_posns: array of pixel (x,y) positions for all
                     lenslets that have fallen on the detector,
                     with shape n_lens_y, n_lens_x, 2
    """

    #define lists of: (1) lenslets to search around
    to_search_around = [0]
    #(2) lenslets that have been searched around
    done_searching_around = []
    #(3) lenslets whose positions have been entered
    positions_entered = [0]
    #(4) lenslet indices organized by positions in the array
    positions = [[1000,1000]]
    #(5) lenslets' pixel positions on the detector,
    #arranged into a (ny,nx,2) array to match the lenslet
    #positions in the array
    posns_pix = np.zeros((2000,2000,2))
    posns_pix[:,:,:] = np.nan
    posns_pix[1000,1000] = avgs[0]

    #(6) lenslets' x,y positions in the lenslet array,
    #plus spot track index in lists of unique spot tracks,
    #arranged into (ny,nx,3) to match lenslet array shape
    #on first and second axis
    posns_idx = np.zeros((2000,2000,3))
    posns_idx[:,:,:] = np.nan
    posns_idx[1000,1000] = [1000,1000,0]

    #create KDTree from list of spot track positions at
    #average wavelength
    kd1 = KDTree(avgs)
    #continue searching until all lenslets have been searched
    while len(done_searching_around) < len(avgs):
        #while a search is needed, go through entries in list
        #of lenslets that have yet to be searched

        # print('CHECK done_searching_around:', len(done_searching_around)) 
        # print('CHECK avgs:', len(avgs))
        # print('CHECK to_search_around:', len(to_search_around))

        for search_lens in to_search_around:
            #confirm that this lenslet is not marked as done
            if search_lens not in done_searching_around:
                #create KDTree from single lenslet to be searched
                kd0 = KDTree(avgs[search_lens:search_lens+1])
                #grab xind,yind for the search lenslet, where xind and
                #yind are the indices of the lenslet in the lenslet array
                xind,yind = np.array(positions)[np.where(np.abs(np.array(positions_entered) - search_lens) < 1e-6)][0]

                #get all neighbors within 23 pixels of the search lenslet
                neighbors = kd0.query_ball_tree(kd1, 23, p=2.0, eps=0)
                #difference the pixel position of the search lenslet
                #with that of its neighbors
                diff = kd1.data[neighbors] - kd0.data
                #loop through list of neighbors
                for ii in range(len(diff[0])):
                    #check that the index has not already been registered
                    if neighbors[0][ii] not in positions_entered:
                        entry = diff[0][ii]
                        #check whether x position is more than 15 pixels greater
                        #than the search lenslet
                        if entry[0] > 15:
                            #this means that the x index is one greater than the
                            #search lenslet
                            xind_new = xind + 1
                            #check whether the y value is less than 15 pixels away from
                            #the search lenslet
                            if abs(entry[1]) < 15:
                                #this means x is greater and y is the same
                                #which means we found the lenslet directly to the right
                                yind_new = yind
                                positions_entered.append(neighbors[0][ii])
                                to_search_around.append(neighbors[0][ii])
                                positions.append([xind_new,yind_new])
                                posns_pix[yind_new,xind_new] = kd1.data[neighbors][0,ii]
                                posns_idx[yind_new,xind_new] = [xind_new,yind_new,neighbors[0][ii]]
                        #check whether the x position is more than 15 pixels less
                        #than the search lenslet
                        elif entry[0] < -15:
                            #this means that the x index is one less than the
                            #search lenslet
                            xind_new = xind - 1
                            #check whether the y value is less than 15 pixels away from
                            #the search lenslet
                            if abs(entry[1]) < 15:
                                #this means x is greater and y is the same
                                #which means we found the lenslet directly to the left
                                yind_new = yind
                                posns_pix[yind_new,xind_new] = kd1.data[neighbors][0,ii]
                                posns_idx[yind_new,xind_new] = [xind_new,yind_new,neighbors[0][ii]]
                                positions_entered.append(neighbors[0][ii])
                                to_search_around.append(neighbors[0][ii])
                                positions.append([xind_new,yind_new])
                        elif entry[1] > 15:
                            yind_new = yind+1
                            xind_new = xind
                            #if yind_new > posns.shape[0]:
                            #    zeros_row = np.zeros((1,posns.shape[1],posns.shape[2]))
                            #    posns = np.vstack((posns,zeros_row))
                            posns_pix[yind_new,xind_new] = kd1.data[neighbors][0,ii]
                            posns_idx[yind_new,xind_new] = [xind_new,yind_new,neighbors[0][ii]]
                            positions_entered.append(neighbors[0][ii])
                            to_search_around.append(neighbors[0][ii])
                            positions.append([xind_new,yind_new])
                            #done.append(neighbors[0][ii])

                        elif entry[1] < -15:
                            #print('found lenslet below')
                            xind_new = xind
                            yind_new = yind-1
                            #if yind_new < 0:
                            #    zeros_row = np.zeros((1,posns.shape[1],posns.shape[2]))
                            #    posns = np.vstack((zeros_row,posns))
                            posns_pix[yind_new,xind_new] = kd1.data[neighbors][0,ii]
                            posns_idx[yind_new,xind_new] = [xind_new,yind_new,neighbors[0][ii]]
                            positions_entered.append(neighbors[0][ii])
                            to_search_around.append(neighbors[0][ii])
                            positions.append([xind_new,yind_new])
                            #done.append(neighbors[0][ii])
                        elif (abs(entry[0]) < 15) and (abs(entry[1]) < 15):
                            print('found search lenslet - do nothing!')
                        else:
                            print('uh oh didnt find a neighbor!')
                            stop
                done_searching_around.append(search_lens)



    minx = np.nanmin(posns_idx[:,:,0])
    maxx = np.nanmax(posns_idx[:,:,0])

    miny = np.nanmin(posns_idx[:,:,1])
    maxy = np.nanmax(posns_idx[:,:,1])

    posns_idx[:,:,0]-=minx
    posns_idx[:,:,1]-=miny

    final_posns = np.zeros([int(maxy),int(maxx)])
    final_posns = posns_idx[int(miny):int(maxy+1),int(minx):int(maxx+1)]

    if show_plots==True:
        dists = np.sqrt(posns_idx[:,:,0]**2 + posns_idx[:,:,1]**2)
        plt.imshow(dists)
        plt.colorbar()
        plt.show()

        f = plt.figure(figsize=(11,5))
        f.add_subplot(121)
        plt.title('L band: lenslet x positions\n'+'(in lenslet array, total='+str(int(np.nanmax(final_posns[:,:,0])+1))+')')
        plt.imshow(final_posns[:,:,0])
        plt.colorbar()
        f.add_subplot(122)
        plt.title('L band: lenslet y positions\n'+'(in lenslet array, total='+str(int(np.nanmax(final_posns[:,:,1])+1))+')')
        plt.imshow(final_posns[:,:,1])
        plt.colorbar()
        plt.savefig('Lnew_lens_reg.png',dpi=300)
        plt.show()
    return final_posns

def gen_rectmat_inds(calims,posarr):

    """
    Function to generate row and column indices for sparse matrix
    for all 108 x 108 lenslets and wavelengths.
    """

    matrowinds = []
    matcolinds = []
    matvals = []

    for ll in range(len(calims)):
        for lensx in range(posarr.shape[2]):
            for lensy in range(posarr.shape[1]):
                xc,xs,xe,yc,ys,ye = posarr[ll,lensy,lensx]
                if np.isnan(xc)==False:
                    xc = int(xc)
                    yc = int(yc)
                    xs = int(xs)
                    xe = int(xe)
                    ys = int(ys)
                    ye = int(ye)
                    flatinds = gen_sparse_inds(xs,ys,xe,ye)
                    vals = crop_sparse_vals(calims[ll],xs,xe,ys,ye)
                    for i in range(len(vals)):
                        if vals[i] > 0:
                        #matvals.append(vals[i])
                            matvals.append(1.0)
                            matcolinds.append(flatinds[i])
                            matrowinds.append(lensx+lensy*posarr.shape[2]+ll*posarr.shape[1]*posarr.shape[2])
    return matrowinds, matcolinds, matvals

def gen_sparse_inds(xs,ys,xe,ye,ypix=2048,xpix=2048):
    """
    Function to take 2d x,y pixel coordinates and turn them into flattened
    coordinates for sparse matrix construction.
    """

    indsx = np.array([xval for xval in range(xs,xe) for yval in range(ys,ye)])
    indsy = np.array([yval for xval in range(xs,xe) for yval in range(ys,ye)])

    flatinds = np.ravel_multi_index((indsy,indsx),(ypix,xpix))
    return flatinds

def crop_sparse_vals(image,xs,xe,ys,ye,cut=0.25):
    """
    Function to crop lenslet PSFs down and then only select pixels above
    a certain flux threshold
    """
    cropped = image[ys:ye,xs:xe]
    cropped = cropped-np.median(cropped)
    cropped[np.where(cropped < cut*np.max(cropped))]=0
    cropped/=np.sum(cropped)
    vals = np.array([cropped[yind,xind] for xind in range(0,xe-xs) for yind in range(0,ye-ys)])
    return vals

def gen_QL_rectmat(calims,posarr):
    """
    Function to generate rectmat from cube of cal unit images.
    """

    matrowinds,matcolinds,matvals = gen_rectmat_inds(calims,posarr)
    rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(np.prod(posarr.shape[:3]),np.prod(calims[0].shape)))
    return rmat



def gen_c2_rectmat_inds(calims,posarr):

    """
    Function to generate row and column indices for sparse matrix
    for all 108 x 108 lenslets and wavelengths.
    """

    matrowinds = []
    matcolinds = []
    matvals = []
    #testrow = np.zeros([2048*2048,2])
    #posarr contains 54, 108, 108, 6 for [xc,xs,xe,yc,ys,ye]
    for ll in range(len(calims)):
        for lensx in range(posarr.shape[2]):
            for lensy in range(posarr.shape[1]):
                xc,xs,xe,yc,ys,ye = posarr[ll,lensy,lensx]
                if np.isnan(xc)==False:
                    xc = int(xc)
                    yc = int(yc)
                    xs = int(xs)
                    xe = int(xe)
                    ys = int(ys)
                    ye = int(ye)
                    flatinds = gen_sparse_inds(xs,ys,xe,ye)
                    vals = crop_sparse_vals(calims[ll],xs,xe,ys,ye)
                    for i in range(len(vals)):
                        if vals[i] > 0:
                            matvals.append(vals[i])
                            matrowinds.append(flatinds[i])
                            matcolinds.append(lensx+lensy*posarr.shape[2]+ll*posarr.shape[1]*posarr.shape[2])
    return matrowinds, matcolinds, matvals



def gen_C2_rectmat(calims,posarr):
    """
    Function to generate rectmat from cube of cal unit images.
    """

    print('doing c2 rectmat lowres')
    matrowinds,matcolinds,matvals = gen_c2_rectmat_inds(calims,posarr)
    print(len(matrowinds),len(matcolinds),len(matvals))
    rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(np.prod(calims[0].shape),np.prod(posarr.shape[:3])))
    return rmat





def find_all_spots(ims_cal, lams_u, min_distance=15, thresh=50, plot_im=False):
    spots = {}

    for ii in range(len(ims_cal)):
        im_cal = ims_cal[ii]
        data_smooth = gaussian_filter(im_cal, sigma=0.8)

        coords_yx = peak_local_max(
            data_smooth,
            min_distance=min_distance,
            threshold_abs=np.percentile(data_smooth, thresh),
            exclude_border=5
        )

        print(f"Found {len(coords_yx)} spots in image {ii}")

        coordinates = np.column_stack([coords_yx[:, 1], coords_yx[:, 0]])
        intensities = data_smooth[coords_yx[:, 0], coords_yx[:, 1]]

        spots[ii] = {
                'filename': 'blank',
                'coordinates': coordinates,
                'intensities': intensities,
                'lam': lams_u[ii]*np.ones(len(coordinates)),
                'n_spots': len(coordinates)
            }
        if plot_im == True:
            f = plt.figure(figsize=(20,10))
            f.add_subplot(121)
            plt.imshow(ims_cal[ii])
            plt.scatter(coordinates[:,0],coordinates[:,1],c='r',s=2)
            plt.title('spots in image '+str(ii))
            f.add_subplot(122)
            plt.hist(intensities,bins=100)
            plt.title('spot intensities')
            plt.show()

    return spots


def find_avg_spotpos(spot_tracks_u,lmin,lmax,show_plots=False):
    avgs = []
    for j in range(len(spot_tracks_u)):
        keys = list(spot_tracks_u[j].keys())
        x0,y0 = spot_tracks_u[j][keys[0]][0],spot_tracks_u[j][keys[0]][1]
        lam0 = spot_tracks_u[j][keys[0]][3]
        chx = 1
        chy = 1
        lmin = 2.9
        lmax = 4.15
        lam = 0.5*(lmin+lmax)
        xj,yj = get_trace_pos(lam,x0,y0,lam0,chx,chy,lmin,lmax)
        avgs.append([xj,yj])
    avgs = np.array(avgs)

    if show_plots==True:
        plt.scatter(avgs[:,0],avgs[:,1])
        plt.show()

        np.save('/Users/isabelkain/Desktop/SCALES/scales-data-analysis/avgs.npy', avgs)

        # plt.hist(avgs[:,1],bins=1000)
        # plt.show()

        # plt.hist(avgs[:,0],bins=1000)
        # plt.show()
    return avgs


def make_posarr(ims_cal,final_posns,spot_tracks_u,show_plots=False):
    posarr = np.zeros([len(ims_cal),final_posns.shape[0],final_posns.shape[1],6])
    posarr[:,:,:,:] = np.nan
    for i in range(final_posns.shape[0]):
        for j in range(final_posns.shape[1]):
            xpos,ypos,lind = final_posns[i,j]
            if np.isnan(xpos)==False:
                tofill = list(spot_tracks_u[lind].keys())
                for k in tofill:
                    x,y = spot_tracks_u[lind][k][:2]
                    #print(x,y)
                    size = 16
                    xs = np.max([0,int(x-size/2)])
                    xe = np.min([int(x+size/2),len(ims_cal[k])])
                    ys = np.max([0,int(y-size/2)])
                    ye = np.min([int(y+size/2),len(ims_cal[k])])
                    posarr[k,i,j] = [x,xs,xe,y,ys,ye]

    if show_plots==True:
        for i in range(20,40):
            for j in range(40,60):
                plt.scatter(posarr[:,i,j,0],posarr[:,i,j,3],c=range(len(posarr)))
        plt.show()
    return posarr

