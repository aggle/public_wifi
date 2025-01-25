#! /usr/bin/env python

import os
from copy import copy
from typing import Concatenate
import numpy as np

def rclip(x, xmin, xmax):
    dum = x
    if dum<xmin:
        dum = xmin
    elif dum>xmax:
        dum = xmax

    return dum


#=====================================================
def checkbox(data, box):
    
    ''' 
    This function performs the coarse centroiding on the data array provided.
    
    - data:         a 2D array
    - box:          the size over which each element of the checkbox image will be computed. has to be smaller than the size of data
    - bgcorr:       inherit the background correction parameter so can apply here.
    
    '''
    
    #print('performing coarse centroiding on an array of size {0}'.format(np.shape(data)))

    hbox = int(np.floor(box/2.))
    #print hbox
    psum = 0.
    ix = 0.
    iy = 0.
    
    x = hbox + np.arange(np.size(data, axis=1)-box)
    y = hbox + np.arange(np.size(data, axis=0)-box)
    if (box == 1.):
        chbox = np.copy(data)
        maxpx = np.argmax(chbox)
        ix, iy = np.unravel_index(maxpx, np.shape(data))    
    else:
        chbox = np.zeros_like(data)
        for i in x:
            for j in y:
                # remember to add 1 so python sums all 5 pixels
                psumu = np.sum(data[int(j)-hbox:int(j)+hbox+1, int(i)-hbox:int(i)+hbox+1])
                chbox[j,i] = psumu
                if (psumu > psum):
                    ix = i
                    iy = j
                    psum = psumu
    
    #plt.figure()
    #plt.imshow(chbox, origin='lower', aspect='equal', cmap='gist_heat', interpolation='None')
    #plt.plot(iy, ix, marker='x', mew=2., ms = 10.)
    #plt.title('Checkbox image with coarse centroid')
    #plt.show()
    
    
    return ix, iy
    
    

#=====================================================
def fine_centroid(data, cwin, xc, yc):
    
    sumx = 0.
    sumy = 0.
    sump = 0.
    
    # define rwin, half the cwin setting
    rwin = np.floor(cwin/2.)
    
    # remember to add 1 so we get all cwin pixels
    x = (np.round(xc) - np.floor(cwin/2.)) + np.arange(cwin)
    y = (np.round(yc) - np.floor(cwin/2.)) + np.arange(cwin) 
    
 
    
    for i in x:
        for j in y:
            wx = rclip(rwin-abs(i-xc)+0.5, 0.0, 1.0)
            wy = rclip(rwin-abs(j-yc)+0.5, 0.0, 1.0)
            ww = wx*wy
            
            sumx += ww * data[int(j),int(i)] * i
            sumy += ww * data[int(j),int(i)] * j
            sump += ww * data[int(j),int(i)]
    
    
    # plot of pixel weights, for testing
    #plt.figure()
    #plt.imshow(wtarr[xc-5:xc+5, yc-5:yc+5], origin='lower', aspect='equal', interpolation='None')
    #plt.colorbar()
    #plt.show()
    # end test plot
    
    xc_old = xc
    yc_old = yc
    
    xc = sumx/sump
    yc = sumy/sump

    return xc, yc
#=====================================================
def bgrsub(data, val, size, coord, silent=False):
    
    '''
    Does the background correction step, if needed. Method is determined from the val parameter:
    
    - val <= 0:     no background subtraction (shouldn't come to this function, but let's check anyway)
    - 0 < val < 1:  fractional background. sorts all pixels and subtracts the value of this percentile
    - val > 1:      this value is subtracted uniformly from all pixels
    
    - size:         specifies the size of the pixel area to be used for the calculation. if this number is negative, use the full image array.
    - coord:        inherits the input coordinate from centroid(); required if you provide an roi size
    
    '''
    
 
    if (val <= 0.):
        if silent == False:
            print('No background subtracted')
        outdata = data
    elif (val > 0.) & (val < 1.):
        
        if size < 0:
            # if size is negative, use the full array
            subdata = data

        else:
            subdata = data[np.round(coord[1]-(size/2.)).astype(int):np.round(coord[1]+(size/2.)).astype(int),
                        np.round(coord[0]-(size/2.)).astype(int):np.round(coord[0]+(size/2.)).astype(int)]
            
        bgrval = np.percentile(subdata, val*100.)
        
        # Subtract background level from the FULL image
        outdata = data - bgrval
        if silent == False:
            print('subtracting {0} from image'.format(bgrval))

    else:
        outdata = data - val
    
    return outdata
    
    
#=====================================================

def compute_centroid(
        im : np.ndarray,
        cbox : int = 5,
        cwin : int = 5,
        incoord : tuple[float, float] = (0., 0.),
        bgcorr : float  = -1,
        thresh : float = 0.01,
        silent : bool = False,
) -> tuple[float, float] :
    """
    Implementation of the JWST GENTALOCATE algorithm. Parameters key:

    Parameters
    ----------
    - im:           2-D numpy array with a source located near `incoord`
    - cbox:         the FULL size of the checkbox, in pixels, for coarse centroiding (default = 5)
    - cwin:         the FULL size of the centroid window, in pixels, for fine centroiding (default = 5)
    - incoord:      (x,y) input coordinates of the source position
    - roi:          size of a region of interest to be used for the centroiding (optional). If not set, full image will be used for coarse                       centroiding. 
                        * setting an ROI also requires input coordinates
                        * the ROI size must be bigger than the cbox parameter
    - bgcorr:       background correction parameter. set to:
                        * negative value for NO background subtraction (default)
                        * 0 < bgcorr < 1 for fractional background subtraction
                        * bgcorr > 1 for constant background subtraction number (this number will be subtracted from the entire image)
    - flat:         enter a filename if you have a flat-fielding image to perform flat-fielding
    - out:          enter a filename for output of the fit results to a file (default = None) (not actually used)
    - thresh:       the fit threshold, in pixels. default is 0.1 px. consider setting this to a higher number for testing, long-wavelength
                       data or low SNR data to prevent.
    - silent:       set to True if you want to suppress verbose output

    Output
    ------
    Define your output

    """
    # Do background correction first
    if bgcorr > 0.:
        im = bgrsub(im, bgcorr, -1, incoord, silent=silent)

    ndim = np.ndim(im)

    n = [np.size(im, axis=i) for i in range(ndim)]

    # NOTE: in python the x-coord is axis 1, y-coord is axis 0
    xin = incoord[0]
    yin = incoord[1]
    if silent == False:
        print('Input coordinates = ({0}, {1})'.format(xin, yin))

    roi_im = im
    xoffset = 0
    yoffset = 0
    # Perform coarse centroiding. Pay attention to coordinate
    # offsets
    xc, yc = checkbox(roi_im, cbox)
    xc += xoffset
    yc += yoffset
    if silent == False:
        print('Coarse centroid found at ({0}, {1})'.format(xc, yc))
    # Iterate fine centroiding
    # Take the threshold from the input parameter thresh
    iter_thresh = thresh
    nconv = 0
    while nconv == 0:
        xf, yf = fine_centroid(im, cwin, xc, yc)
        err = np.sqrt((xf-xin)**2 + (yf-yin)**2)
        if silent == False:
            print(("Fine centroid found at (x, y) = ({0:.4f}, {1:.4f}). "
               "Rms error = {2:.4f}".format(xf, yf, err)))
        if (abs(xf-xc) <= iter_thresh) & (abs(yf-yc) <= iter_thresh):
            nconv = 1
        xc = xf
        yc = yf

    return xf, yf


#======================================================
def fixbadpix(data, maxstampwidth=3, method='median'):
    """
    Replace the values of bad pixels in the TA image
    with interpolated values from neighboring good
    pixels. Bad pixels are identified as those with a
    value of 65535 in the flat field file

    Parameters:
    -----------
    data -- 2D array containing the TA image
    bpix -- Tuple of lists of bad pixel coordinates. 
            (output from np.where on the 2D flat field image)
    maxstampwidth -- Maximum width of area centered on a bad pixel
                     to use when calculating median to fix the bad 
                     pixel. Must be an odd integer.
    method -- The bad pixel is fixed by taking the median or the 
              mean of the surrounding pixels. This is a string
              that can be either 'median' or 'mean'.

    Returns:
    --------
    2D image with fixed bad pixels
    """
    ny, nx = data.shape

    # Set up the requested calculation method
    if method == "median":
        mmethod = np.nanmedian
    elif method == "mean":
        mmethod = np.nanmean
    else:
        print("Invalid method. Must be either 'median' or 'mean'.")
        sys.exit()
    
    if (maxstampwidth % 2) == 0:
        print("maxstampwidth must be odd. Adding one to input value.")
        maxstampwidth += 1 

    half = np.int((maxstampwidth - 1)/2)

    bpix = np.isnan(data)
    bad = np.where(bpix)
    # Loop over the bad pixels and correct
    for bady, badx in zip(bad[0], bad[1]):

        print('Bad pixel:',bady,badx)
        
        substamp = np.zeros((maxstampwidth, maxstampwidth))
        substamp[:,:] = np.nan
        minx = badx - half
        maxx = badx + half + 1
        miny = bady - half
        maxy = bady + half + 1

        # Offset between data coordinates and stamp
        # coordinates
        dx = copy(minx)
        dy = copy(miny)

        # Check for stamps that fall off the edges
        # of the data array
        sminx = 0
        sminy = 0
        smaxx = maxstampwidth
        smaxy = maxstampwidth
        if minx < 0:
            sminx = 0 - minx
            minx = 0
        if miny < 0:
            sminy = 0 - miny
            miny = 0
        if maxx > nx:
            smaxx = maxstampwidth - (maxx - nx)
            maxx = nx
        if maxy > ny:
            smaxy = maxstampwidth - (maxy - ny)
            maxy = ny
            
        substamp[sminy:smaxy, sminx:smaxx] = data[miny:maxy, minx:maxx]

        # First try the mean of only the 4 adjacent pixels
        neighborsx = [half, half+1, half, half-1]
        neighborsy = [half+1, half, half-1, half]
        if np.sum(np.isnan(substamp[neighborsx, neighborsy])) < 4:
            data[bady, badx] = mmethod(substamp[neighborsx, neighborsy])
            print(("Good pixels within nearest 4 neighbors. Mean: {}"
                   .format(mmethod(substamp[neighborsx, neighborsy]))))
            continue

        # If the adjacent pixels are all NaN, expand to include corners
        else:
            neighborsx.append([half-1, half+1, half+1, half-1])
            neighborsy.append([half+1, half+1, half-1, half-1])
            if np.sum(np.isnan(substamp[neighborsx, neighborsy])) < 8:
                data[bady, badx] = mmethod(substamp[neighborsx, neighborsy])
                print(("Good pixels within 8 nearest neighbors. Mean: {}"
                       .format(mmethod(substamp[neighborsx, neighborsy]))))
                continue

        # If all pixels are still NaN, iteratviely expand to include
        # more rings of pixels until the entire stamp image is used
        # (This step not included in Goudfrooij's original bad pixel
        # correction script).
        delta = 2
        while delta <= half:
            newy = np.arange(half-(delta-1), half+delta)
            newx = np.repeat(half - delta, len(newy))
            neighborsx.extend(newx)
            neighborsy.extend(newy)
            newx = np.repeat(half + delta, len(newy))
            neighborsx.extend(newx)
            neighborsy.extend(newy)
            newx = np.arange(half-delta, half+delta+1)
            newy = np.repeat(half - delta, len(newx))
            neighborsx.extend(newx)
            neighborsy.extend(newy)
            newy = np.repeat(half + delta, len(newx))
            neighborsx.extend(newx)
            neighborsy.extend(newy)
            if np.sum(np.isnan(substamp[neighborsx, neighborsy])) < (len(neighbosrsx)):
                data[bady, badx] = mmethod(substamp[neighborsx, neighborsy])
                print("Expanding to {} rows".format(delta))
                continue
            else:
                delta += 1
        print(("Warning: all pixels within {} rows/cols of the bad pixel at ({},{}) "
               "are also bad. Cannot correct this bad pixel with this stamp image"
               "size.".format(delta, badx, bady)))

    return data

        
#====================================================== 
