# Method for bootstrapping a/Rs and cos i variables, given an impact parameter, transit duration, radius ratio, and orbital period, with 1-sigma error ba_Rs
# Author: Hannah Diamond-Lowe
# Date: 24 July 2014
# Updated: 11 March 2016    now a class!

import numpy as np
import matplotlib.pyplot as plt

class Bootstrap(object):

    def __init__(self, b, b_unc, Tdur, Tdur_unc, RpRs, RpRs_unc, P):
        self.b = b
        self.b_unc = b_unc
        self.Tdur = Tdur
        self.Tdur_unc = Tdur_unc
        self.RpRs = RpRs
        self.RpRs_unc = RpRs_unc
        self.P = P
        self.iterations = 100000 # number of iterations to perform; increase until you get consistent histograms.
        # create distributions
        # np.random.normal(mean, sigma, number of points) --> creates a gaussian distribution centered on the mean, with 1-sigma as sigma
        self.b_dist = np.random.normal(self.b,self.b_unc,self.iterations) # impact parameter b = a/Rs * cos(i)
        self.Tdur_dist = np.random.normal(self.Tdur,self.Tdur_unc,self.iterations) # transit duration (days); this duration is from T1.5 - T3.5, from the center of the planet hitting the limb of the star
        self.RpRs_dist = np.random.normal(self.RpRs,self.RpRs_unc,self.iterations) # Rp/Rs
        # create initial guess
        self.guess = 1.0   # starting guess for the dependence a/Rs

    def get_a_Rs(self):
        self.a_Rs = np.random.normal(100, 10, self.iterations)  # some random distribution (with starting mean is far from guess)
        # run iterations until the maximum difference between the guess and one of the a/Rs distribution values is less than 10e-4
        while abs(self.guess - self.a_Rs).max() > 10e-4:
            self.guess = self.a_Rs    
            # a/Rs equation derived from Tdur = 0.5(T14 - T23) and rearranged to solve for one of the a/Rs values in terms of the other
            self.a_Rs = (((1 + self.RpRs)**2 - self.b**2)*(np.sin(2*np.pi/self.P*self.Tdur - np.arcsin(np.sqrt(((1-self.RpRs)**2 - self.b**2)/(self.guess**2 - self.b**2))))**(-2)) + self.b**2)**(0.5)
        return np.mean(self.a_Rs)

    def plot_a_Rs(self):
        # plot a histrogram of resulting a/Rs values; check for Gaussian distribution
        plt.hist(self.a_Rs, bins=50)
        plt.title("a/Rs histogram")
        plt.show()

    def get_cosi(self):
        self.cosi = self.b/self.a_Rs
        return np.mean(self.cosi)

    def plot_cosi(self):
        # plot a histrogram of resulting cos(i) values; check for Gaussian distribution
        plt.hist(self.cosi, bins = 50)
        plt.title("cos(i) histogram")
        plt.show()
