import numpy as np

class UncertaintyPredictor(object):

    def __init__(self, ref_name, ref_mag, ref_texp, ref_ppm, targ_name, targ_mag, targ_dur, inst_readout):

        self.ref_name     = ref_name
        self.ref_mag      = ref_mag          # magnitude of literature reference data
        self.ref_texp     = ref_texp         # exposure time [s]
        self.ref_ppm      = ref_ppm          # [ppm]
        self.targ_name    = targ_name
        self.targ_mag     = targ_mag         # magnitude of target (must be same band as reference magnitude)
        self.targ_dur     = targ_dur         # duration of target transit [s]
        self.inst_readout = inst_readout     # read out time of instrument [s]

    def times_fainter(self, ref_mag, targ_mag):
        Dmag = (targ_mag - ref_mag)
        return np.power(10., (0.4*Dmag))

    def Nphot_ref(self, ppm):
        return 1./((ppm*1e-6)**2)

    def Nphot_targ(self, ref_phot, timesfainter):
        return ref_phot/timesfainter

    def texp_samecadence(self, ref_texp, ref_Nphot, targ_Nphot):
        return (ref_Nphot/targ_Nphot)*ref_texp

    def texp_sigma(self, texp, ref_texp, targ_Nphot):
        Nphot = targ_Nphot*(texp/ref_texp)
        sigma = 1./np.sqrt(Nphot)
        return sigma

    def duration_sigma(self, readout, texp, duration, sigma):
        cadence = readout + texp
        Nexps = duration/cadence
        print('cadence is {0} seconds; {1} exposures per transit duration'.format(cadence, Nexps))
        return sigma/np.sqrt(Nexps)

    def exposure_times(self, test_times):
        self.test_times = test_times    # array of exposure times to test [s]
        self.ref_Nphot = self.Nphot_ref(self.ref_ppm)
        self.targ_timesfainter = self.times_fainter(self.ref_mag, self.targ_mag)
        self.targ_Nphot = self.Nphot_targ(self.ref_Nphot, self.targ_timesfainter)

        print('using the same exposure time as reference star ({0} seconds),'.format(self.ref_texp))
        print('exposure time needed to collect same number of photons as reference [s] = ' + str(self.texp_samecadence(self.ref_texp, self.ref_Nphot, self.targ_Nphot)))
        print('')

        for t in self.test_times:
            texp_sigma = self.texp_sigma(t, self.ref_texp, self.targ_Nphot)
            dur_sigma = self.duration_sigma(self.inst_readout, t, self.targ_dur, texp_sigma)
            print('using {0} s exposures for {1}, sigma = {2} per exposure'.format(str(t), self.targ_name, texp_sigma))
            print('{0} transit depth uncertainty = {1} per transit ({2} s exp)\n'.format(self.targ_name, dur_sigma, t))

    def scale_height(self, Temp, mu, g):		
        # takes effective temperature [K], mu is weighted average of molecules, g is surface gravity in m/s^2
        k  = 1.38e-23		# Boltzmann constant [J/K]
        mH = 1.67e-27	    # mass of hydrogen atom [kg]
        return k*Temp/(mu*mH*g)/1000.	# [km]

    def Delta_transit_depth(self, H, numH, Rp, Rs):       
        # takes scale height [km], number of scale heights, planet radius [km], stellar radius [km]
        return ((Rp + numH*H)/Rs)**2 - (Rp/Rs)**2

    def atmosphere_params_init(self, targ_Rs, targ_Rp, targ_T, targ_g, molecule_name, mu):

        RE = 6317.          # radius of earth [km]
        RS = 6.963e5        # radius of Sun [km]
        self.targ_Rs       = targ_Rs*RS     # radius of target star [km]
        self.targ_Rp       = targ_Rp*RE     # radius of target planet [km]
        self.targ_T        = targ_T         # equilibrium temperature of planet [K]
        self.targ_g        = targ_g         # surface gravity of planet [m/s^2]
        self.molecule_name = molecule_name   # name of dominant atmospheric species
        self.mu            = mu             # mean molecular weight of dominant atmosphere species

        self.H = self.scale_height(self.targ_T, self.mu, self.targ_g)

    def atmosphere_detection(self, num_H, num_transits, texp):

        print('to detect {0} atmosphere on {1}:'.format(self.molecule_name, self.targ_name))
        for ntr in num_transits:        
            for nh in num_H:
                Ddepth = self.Delta_transit_depth(self.H, nh, self.targ_Rp, self.targ_Rs)
                texp_sigma = self.texp_sigma(texp, self.ref_texp, self.targ_Nphot)
                dur_sigma = self.duration_sigma(self.inst_readout, texp, self.targ_dur, texp_sigma)
                print('change in transit depth for {0} H = {1}'.format(str(nh), str(Ddepth)))
                print('detect {0} atmosphere at {1} sigma'.format(self.molecule_name, str(Ddepth/dur_sigma)))
                print('with {0} transits, detect {1} {2} atmosphere at {3} sigma'.format(str(ntr), str(nh), self.molecule_name, str(Ddepth/(dur_sigma/np.sqrt(ntr)))))
                print('')
