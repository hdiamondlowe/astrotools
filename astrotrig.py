# functions that will help when working with astronomy units
# Author: Hannah Diamond-Lowe
# Date: 1 Mar 2016

import numpy as np

# some useful little funcitons
def cos(deg):
    # cosine function that takes degrees
    return np.cos(np.radians(deg))

def sin(deg):
    # sine function that takes degees
    return np.sin(np.radians(deg))

def arcsec_to_deg(arcsec):
    # converts arcseconds to degrees
    return arcsec/60./60.

def arcsec_to_sec(arcsec):
    # converts arcseconds to seconds
    return arcsec/15.

def sec_to_arcsec(sec):
    # converts seconds to arcseconds
    return sec*15.
