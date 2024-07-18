# code to simulate binned spectral data
# Date: 8 Nov. 2015
# Author: Hannah Diamond-Lowe
# Update: 4 Oct. 2016   This has been adapted to work with Eliza Kempton's open source Exo_Transmit code
# Update: 14 Jan. 2022  This has been adapted to work with Joao Mendonca's GoldenRetriever code (beta version) 


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import h5py
from astropy.convolution import convolve, Gaussian1DKernel

"""
function takes
    datafile (name of data file); table should be formated as it comes out of Exo_Transmit: two columns, no headers, col1 = wavelengths [m], col2 = transit depth (Rp/Rs)^2 [%]  --> these will be converted to wavelengths [microns] and transit depths (Rp/Rs)^2  []
    binsize [microns]
    wavelength_range [microns]; two column list: [low, high]
    smooth [microns]; (optional) if you want to smooth the spectrum to the given micron segment (protip: make this smaller than the binsize input)

function returns
    wavelengths     only wavelengths of wihtin the givne range
    depths          model depth at every wavelength in wavelengths
    bin_midpoints   wavelengths in the center of each bin; binsize given as input
    bin_avgs        the model-average depth in each bin, corresponding to bin_midpoints
"""
def model_spectrum(datafile, binsize, wavelength_range, smooth=False, model_code='ExoTransmit'):

    # retrieve data, column of wavelengts, column of depths
    if model_code == 'ExoTransmit': 
        data = ascii.read(datafile) # this is currently how Eliza's custom TP profiles come out; double check this is what you want
        try:
            wavelengths = data['Wavelength']*1e6 # convert from meters to microns
            depths = data['Transit Depth']/1e2   # convert from percentage to true value
        except(KeyError): 
            wavelengths = data['col1']*1e6 # convert from meters to microns
            depths = data['col2']/1e2   # convert from percentage to true value

    elif model_code == 'GoldenRetriever':
        with h5py.File(datafile, 'r') as file:
            wavelengths = np.array([k for k in file['wavelength'][:]])
            depths = np.array([k for k in file['spectrum'][:]])

    inds = np.where((wavelengths >= wavelength_range[0]) & (wavelengths <= wavelength_range[1]))
    wavelengths = wavelengths[inds]
    depths = depths[inds]

    # create lists of wavelength midpoints of each bin and list of average depths in each bin
    bin_midpoints = []
    bin_avgs = []

    # set the beginning and end value of each bins
    #bin_limits = np.arange(wavelengths[0], wavelengths[-1], binsize)
    bin = int(np.round((wavelengths[-1] - wavelengths[0])/binsize))
    bin_limits = np.histogram(wavelengths, bin)[1]

    # get the wavelength midpoint of a bin and take the average depth in this bin; populate lists accordingly
    for i in range(len(bin_limits)-1):
        bin_indexes = np.where((wavelengths >= bin_limits[i]) & (wavelengths <= bin_limits[i+1]))
        if len(bin_indexes[0]) <= 1:
            continue# "need a bigger bin size"
        bin_wavelengths = wavelengths[bin_indexes]
        bin_depth = depths[bin_indexes]
        bin_midpoints.append(np.median([bin_limits[i], bin_limits[i+1]]))
        bin_avgs.append(np.mean(bin_depth))


    # return a 4D list of wavelengths from data, depths from data, wavelength midpoints, and average depth per bin
    if smooth != False:
        #window, order = smooth[0], smooth[1]
        #smooth_depths = savitzky_golay(depths, window, order)
        gauss_kernel = Gaussian1DKernel(smooth[0])
        smooth_depths = convolve(np.array(depths), gauss_kernel, boundary='extend')
        return [wavelengths, smooth_depths, bin_midpoints, bin_avgs]
    else: return [wavelengths, depths, bin_midpoints, bin_avgs]

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except(ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')