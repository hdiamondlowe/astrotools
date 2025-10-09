import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import Table
from astropy.modeling.models import BlackBody
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.time import Time
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
from scipy import stats, interpolate
import pysynphot
import pickle
import json
import copy
import os
import glob
from collections import OrderedDict
import batman

from jwst_backgrounds import jbt
import pandeia.engine
from pandeia.engine.calc_utils import build_default_calc
from pandeia.engine.perform_calculation import perform_calculation
from time import perf_counter

print(pandeia.engine.pandeia_version())

import warnings
warnings.filterwarnings('ignore')

'''
This class is to help make some basic S/N calculations for observing TSOs with MIRI MRS.
It should keep working with updated versions of pandeia-engine.
--- H. Diamond-Lowe 2025-09-24
'''

class MIRIMRS_Observation_TSO:
    """A class to handle MIRI MRS TSO observations for JWST.
    
    """

    def __init__(self, target, eventtype='eclipse',
                 frac_fullwell=0.65, tfrac=1,
                 ngroup_list=[], ngroups_remove=0,
                 stellar_spec='default', input_spec=None, save_spec=False,
                 verbose=True, savepath='./'):
        self.target = target
        self.eventtype = eventtype
        self.frac_fullwell = frac_fullwell
        self.tfrac = tfrac
        self.ngroup_list = ngroup_list
        self.ngroups_remove = ngroups_remove
        self.stellar_spec = stellar_spec
        self.input_spec = input_spec
        self.save_spec = save_spec
        self.verbose = verbose
        self.savepath = savepath

        self.dispersers = ['short', 'medium', 'long']
        self.apertures = ['ch1', 'ch2', 'ch3', 'ch4']
        self.mrs_bands = [f'{disperser}_{aperture}' for disperser in self.dispersers for aperture in self.apertures]
        # Initialize other attributes

        # an estimate of the transit duration is key throughout but requires a lot of parameters
        # if you have these parameters, great, but if you'd like to use your own transit duration
        # (like maybe if this is for an eclispe and the eclispe duration is different than the transit duration)
        # you can do that too if you ad the keyword "pl_tdur"
        #try: self.tdur = self.target['pl_tdur'] * u.day
        #except(KeyError):

        try: self.tdur = self.target['pl_tdur'] * u.day
        except(KeyError):

            if self.target['pl_ratror'] == None:
                print("Calculating Rp/Rs from separate Rp and Rs")
                self.target['pl_ratror'] = ((self.target['pl_rade']*u.R_earth)/(self.target['st_rad']*u.R_sun)).decompose().value
            if self.target['pl_ratdor'] == 0.0:
                print("Calculating a/Rs from separate a and Rs")
                self.target['pl_ratdor'] = ((self.target['pl_orbsmax']*u.AU)/(self.target['st_rad']*u.R_sun)).decompose().value
            if self.target['pl_imppar'] == 0.0 or np.isnan(self.target['pl_imppar']):
                print("Calculating impact parameter from inclination")
                self.target['pl_imppar'] = self.impact(self.target['pl_ratdor'], self.target['pl_orbincl'])
            if self.target['pl_orbincl'] == 0.0 or np.isnan(self.target['pl_orbincl']):
                print("Calculating inclination from impact parameter")
                self.target['pl_orbincl'] = self.inc(self.target['pl_ratdor'], self.target['pl_imppar'])

            # need eclipse duration
            self.tdur = self.calc_Tdur() # event duration
            if np.isnan(self.tdur): 
                print("ERROR: duration cannot be nan")

        if self.verbose: print("Transit duration:", self.tdur.to(u.hr))

    def get_target_timing(self):
        """
        Calculates and stores timing and configuration parameters for a MIRI MRS time-series eclipse observation.
        Determines optimal detector group settings, integration times, cadence, and other relevant parameters for JWST MIRI MRS TSO.
        Uses target and observation parameters, performs ETC calculations, and updates self.timing for each band.
        
        Returns
        -------
        None
            Results are stored in self.timing[band] for each band.
        
        Notes
        -----
        - If self.ngroup_list is provided, uses those values for ngroups; otherwise, calculates based on saturation and full well fraction.
        - Issues warnings for suboptimal group numbers and prints ETC warnings if verbose.
        - Updates self.input_spec with a PHOENIX stellar spectrum if self.stellar_spec is 'default'.
        """
    
        self.timing = {}

        if self.verbose: 
            print(self.target)  

        # obs params specific to MIRI imaging
        tsettle = 30 * u.min    # detector settling time
        tcharge = 1 * u.hr      # amount of charged time becuase JWST will not start observations right away
        mingroups = 5           # suggested for MIRI Imaging TSO
     
        approx_obs_time = (self.tdur + self.tfrac*self.tdur + tsettle + tcharge).to(u.hr)
        if self.verbose: print('Approximate observing time per eclipse observation:', approx_obs_time)

        if self.stellar_spec == 'default':
            Phoenix_spec = pysynphot.Icat('phoenix', Teff=self.target['st_teff'], metallicity=0.0, log_g=self.target['st_logg'])
            Phoenix_spec.convert('flam')
            wave = Phoenix_spec.wave * u.Angstrom
            flux = Phoenix_spec.flux * u.erg/u.cm**2/u.s/u.Angstrom
            flux *= wave**2 / const.c

            self.input_spec = np.array([wave.to(u.um).value, flux.to(u.mJy).value])
            if self.save_spec: ascii.write(self.input_spec.T, f"{self.target['hostname']}_pysynphot_PHOENIX_spec.dat", overwrite=True)
        
        for bnd, band in enumerate(self.mrs_bands):
            # make the pandeia dictionary for miri mrs

            self.timing[band] = {}

            disperser, aperture = band.split('_')
            if self.verbose: print(f'***Working on disperser={disperser}, aperture={aperture}, frac_fullwell={self.frac_fullwell}')

            miri_mrs_ts = self.make_miri_dict(disperser, aperture, scene_spec='input', input_spec=self.input_spec)

            if len(self.ngroup_list) > 0:
                print('You are setting the number of groups. The full well fraction is ignored.')
                self.ngroups = self.ngroup_list[bnd]
            
            else: 
                report = perform_calculation(miri_mrs_ts)
                print('disperser', disperser, 'aperture', aperture, '; ngroups before for saturation:', report['scalar']['sat_ngroups'])
                ngroups = int(np.floor(report['scalar']['sat_ngroups']*self.frac_fullwell))  # use as many groups as possible without saturating, times the fraction of the full well depth
            
            if self.ngroup_list == []:
                if ngroups > 300: 
                    ngroups = 300        # ngroups > 300 not recommended due to cosmic rays ### FIX THIS; I think its 300 seconds, not ngroups
                    print('    WARNING: ngroups is >300, setting to 300; not recommended due to cosmic rays')
                elif ngroups>=5 and ngroups<=10: 
                    print('    WARNING: ngroup is in the 5-10 range --> adding 1 group')
                    ngroups+=1
                elif ngroups < mingroups:
                    print('    WARNING: THIS IS TOO FEW GROUPS!')
                elif ngroups < 20:
                    print(f'    WARNING: ngroups is {ngroups}, less than recommended ~20')
            
                self.ngroups = ngroups

            miri_mrs_ts['configuration']['detector']['ngroup'] = self.ngroups - self.ngroups_remove
            report = perform_calculation(miri_mrs_ts)

            if self.verbose:
                print('    ETC Warnings:')
                print(        report['warnings']) # should be empty if nothing is wrong
            
            tframe  = report['information']['exposure_specification']['tframe'] * u.s
            tint    = tframe * (self.ngroups - self.ngroups_remove)  # amount of time per integration
            treset  = (1+self.ngroups_remove) * tframe                                       # reset time between each integration
            cadence = tint + treset
            nint    = (self.tdur/(tint + treset)).decompose()        # number of in-transit integrations
            ref_wave = report['scalar']['reference_wavelength']                         * u.micron
            
            if self.verbose: 
                print('    Timing for each observation')
                print('        Number of groups per integration (ngroups)', self.ngroups)
                print('        Number of groups removed ', self.ngroups_remove)
                print('        Time to take one frame (tframe)', tframe)
                print('        Time per single integration (tframe*ngroup):', tint)
                print('        Cadence (integration time plus reset):', cadence)
                print('        Number of in-occultation integrations:', nint.decompose())
                print('        Fraction of baseline to in-occultation observing time:', self.tfrac)
                print('        Observing efficiency (%):', (tint/cadence).decompose()*100)

            self.timing[band]['tframe']   = tframe
            self.timing[band]['tint']     = tint
            self.timing[band]['treset']   = treset
            self.timing[band]['cadence']  = cadence
            self.timing[band]['nint']     = nint
            self.timing[band]['ngroups']  = self.ngroups
            self.timing[band]['ngroups_sat'] = report['scalar']['sat_ngroups']
            self.timing[band]['ref_wave'] = ref_wave
            self.timing[band]['report']   = report
            self.timing[band]['miri_mrs_ts'] = miri_mrs_ts

    def get_pandeia_snr(self, nobs=1):
        """
        Calculate the signal-to-noise ratio (SNR) for each MIRI MRS band using Pandeia ETC results.

        This method loops over all bands, extracts the SNR and wavelength arrays from the ETC report,
        and computes uncertainties for single integrations and for multiple observations. It also
        calculates the uncertainty on the eclipse depth, accounting for the number of integrations
        in and out of eclipse. Results are stored in self.snr for later analysis.

        Parameters
        ----------
        nobs : int, optional
            Number of eclipse observations to combine (default: 1).

        Returns
        -------
        None
            Results are stored in self.snr[band] for each band.
        """

        self.nobs = nobs

        self.snr = {}

        for band in self.mrs_bands:

            self.snr[band] = {}

            disperser, aperture = band.split('_')

            report          = self.timing[band]['report']
            nint            = self.timing[band]['nint']
            
            # compute uncertainty in a single measurement
            wave = report['1d']['extracted_flux'][0] * u.micron


            snr = report['1d']['sn'][1]

            if self.verbose: 
                print(f'Single integration info, {disperser}, {aperture}')
                #print('    Avg. Signal +/- Noise =', np.mean(extracted_flux), '+/-', np.mean(extracted_noise))
                #print('    Avg. Signal +/- Noise =', np.mean(signal), '+/-', np.mean(noise_1obs))
                #print('    Avg. Fractional uncertainty,', np.mean(noise_1obs/signal))
                print('    Avg. SNR', np.mean(snr))

            # if the event type is a transit let's remove a little fo that snr to get a more realistic estimate
            # note that this scaling of snr breaks down for deep eclipse depths
            if self.eventtype == 'transit':
                snr *= np.sqrt(1 - self.target['pl_ratror']**2)
                if self.verbose: print('    Adjusted for transit depth, new avg. SNR', np.mean(snr))

            self.snr[band]['wave']       = wave
            self.snr[band]['snr']        = snr

            # uncertainties on single integrations
            err_1obs = 1/snr
            err_1obs_broad = np.sqrt(np.sum(err_1obs**2)/len(wave)**2)
            err_nobs = err_1obs/np.sqrt(self.nobs)
            err_nobs_broad = err_1obs_broad * (1/np.sqrt(self.nobs))
            
            # uncertainties on eclipse depth, taking into account number on integrations in and out of eclipse
            depth_err_1obs = err_1obs * (1/np.sqrt(nint)) * np.sqrt(1 + (1/self.tfrac))
            depth_err_1obs_broad = np.sqrt(np.sum(depth_err_1obs**2)/len(wave)**2)
            depth_err_nobs = depth_err_1obs * (1/np.sqrt(self.nobs))
            depth_err_nobs_broad = depth_err_1obs_broad * (1/np.sqrt(self.nobs))

            self.snr[band]['err_1obs'] = err_1obs
            self.snr[band]['err_1obs_broad'] = err_1obs_broad
            self.snr[band]['err_nobs'] = err_nobs
            self.snr[band]['err_nobs_broad'] = err_nobs_broad

            self.snr[band]['depth_err_1obs'] = depth_err_1obs
            self.snr[band]['depth_err_1obs_broad'] = depth_err_1obs_broad
            self.snr[band]['depth_err_nobs'] = depth_err_nobs
            self.snr[band]['depth_err_nobs_broad'] = depth_err_nobs_broad

    def get_model_depth(self, depth_from_model=False, depth_models=[], convolve_model_val=False,
                       Albedos=[]):
        
        Albedos = np.array(Albedos)
        assert not np.any(Albedos > 1) and not np.any(Albedos<0), "Error: cannot have albedo greater than 1 or less than 0"

        if self.eventtype == 'transit':
            assert depth_from_model == True, "For transit events, depth_from_model must be True"
            assert len(depth_models) > 0, "For transit events, depth_models cannot be empty"

        MIRI_MRS_transmission_dir = './MIRI_MRS_transmission_v4.0/'

        self.models = OrderedDict()

        # FIRST MAKING THIS WORK FOR BROADBAND AND THEN WILL TRY TO DO UNBINNED (FOR CROSS CORRELATION)

        # Model atmospheres: from input models or Teq calculations
        self.depth_from_model = depth_from_model
        wave_range = np.linspace(0.7, 30, 100) * u.micron

        def process_band(model_dict, wave, depth, key=None, T_day=None):

            for band in self.mrs_bands:
            
                disperser, aperture = band.split('_')
                ref_wave = self.timing[band]['ref_wave']
            
                transmission_file = f"{MIRI_MRS_transmission_dir}/MIRI_{aperture.upper()}_{disperser.upper()}_transmission_ETC4.0.fits"
            
                with fits.open(transmission_file) as MRS_transmission:
                    bandpass_wave = MRS_transmission[1].data['WAVELENGTH'] * u.micron
                    bandpass_filt = MRS_transmission[1].data['TRANSMISSION']
            
                inds = (wave.value > bandpass_wave[0].value) & (wave.value < bandpass_wave[-1].value)
            
                wave_band = wave[inds]
                depth_band = depth[inds]

                model_dict.setdefault('wave_band', []).append(wave_band)
                model_dict.setdefault('depth_band', []).append(depth_band)

                avg_band = self.get_bandpass_avg_flux(wave_band, depth_band, bandpass_wave, bandpass_filt)
                model_dict.setdefault('wave_broadband', []).append(ref_wave.value if key else ref_wave)
            
                if key:
                    amp_day = self.calc_FpFs(T_day, ref_wave, self.input_spec)
                    model_dict.setdefault('depth_broadband', []).append(amp_day)
                else:
                    model_dict.setdefault('depth_broadband', []).append(avg_band)

        if depth_from_model:
            for model_name, model in depth_models.items():

                self.models[model_name] = {}
                
                wave = model['wavelength'] * u.micron
                depth = model['depth']
                
                if convolve_model_val:
                    wave, depth = self.convolve_model_spec([wave.value, depth], convolve_model_val)
                    wave = wave * u.micron
                
                self.models[model_name]['wave'] = wave
                self.models[model_name]['depth'] = depth

                process_band(self.models[model_name], wave, depth)

        for A in Albedos:
            if self.eventtype == 'transit':
                print("We don't calculate Fp/Fs for transits")
                print("Ignoring the albedo parameter")
                break  # Exit the entire albedo loop

            for case in ['bare_rock', 'equilibrium']:

                T_day = self.calc_Tday(A, atmo=case)
                Fp_Fs = self.calc_FpFs(T_day, wave_range, self.input_spec)

                key = f"{case}_{A}A"
                self.models[key] = {'wave': wave_range, 'fpfs': Fp_Fs, 'T_day': T_day}
                
                process_band(self.models[key], wave_range, Fp_Fs, key=key, T_day=T_day)
                
                if self.verbose: print(case, ':', T_day)

    def simulate_mrs_obs(self, measurement_case=None):
        
        """
        Simulate JWST MIRI MRS observations for the target using model planet/star flux ratios.
        This function generates synthetic light curves and spectra for each model scenario (e.g., bare rock, equilibrium, or user-provided),
        adds noise based on instrument SNR, and produces binned simulated data for analysis.
        It selects a measurement case, injects noise, bins the simulated light curve, and stores results for each band and model.
        The output is stored in self.simdata for further analysis or plotting.
        """

        if self.eventtype =='transit':
            tstart = (self.target['pl_orbper']*u.day) - (self.tdur/2) - (self.tdur*self.tfrac/2)
            tend   = (self.target['pl_orbper']*u.day) + (self.tdur/2) + (self.tdur*self.tfrac/2)
        elif self.eventtype=='eclipse':
            tstart = (self.target['pl_orbper']*u.day)*0.5 - (self.tdur/2) - (self.tdur*self.tfrac/2)
            tend   = (self.target['pl_orbper']*u.day)*0.5 + (self.tdur/2) + (self.tdur*self.tfrac/2)

        trange = tend - tstart

        self.tstart = tstart
        self.tend = tend
        self.trange = trange
        
        # use batman to make the system; 
        batman_params = self.initialize_batman_model()
        self.batman_params = batman_params
        
        def get_binned_signal(time, ts_scatter_nobs):
            bins = 12
            ts_scatter_binned, time_bin_edges, _ = stats.binned_statistic(time, ts_scatter_nobs, 
                                                                          statistic='mean', bins=bins)
            npoints_per_bin, _, _                = stats.binned_statistic(time, ts_scatter_nobs, 
                                                                          statistic='count', bins=bins)
            time_bin_width = np.mean(np.diff(time_bin_edges))
            time_binned = time_bin_edges[:-1] + time_bin_width/2

            return time_binned, ts_scatter_binned, npoints_per_bin

        #else: cases = np.hstack(list(self.models.keys()))
        cases = np.hstack(list(self.models.keys()))

        if measurement_case==None:
            if self.depth_from_model==False: 
                measurement_case = f'{measurement_case}_{Albedos[0]}A'
            else:
                measurement_case = list(self.models.keys())[0]
        else:
            assert measurement_case in self.models.keys(), "Error: measurement_case not in models"

        #draw_eclipse_depth_measurement = np.random.normal(self.models[measurement_case]['depth_band'], yerr_nobs, ndraws)
        #dof = 1  # rough estimate; only 1 data point

        self.simdata = OrderedDict()

        for i, case in enumerate(cases):

            self.simdata[case] = {}
            self.simdata[case]['measurement_case'] = False

            for bnd, band in enumerate(self.mrs_bands):

                np.random.seed()  # Use system entropy for non-deterministic seeding; for reproducibility, use a fixed integer like np.random.seed(42)

                self.simdata[case][band] = {}

                total_int = int(np.ceil((trange/self.timing[band]['cadence']).decompose()))

                # create light curve time points (will use to compute depth)
                time = np.linspace(tstart.value, tend.value, total_int)/self.target['pl_orbper'] # time in phase

                if self.eventtype == 'transit':
                    lc = self.get_batman_lightcurve_tra(time)
                elif self.eventtype == 'eclipse':
                    lc = self.get_batman_lightcurve_ecl(time, self.models[case]['depth_broadband'][bnd])
                self.simdata[case][band]['time'] = time
                self.simdata[case][band]['lc']   = lc

                # resample the model onto the snr wavelengths --- this is the true "simulated data"
                depth_spec_sampled = self.resample_flux_conserving(self.models[case]['wave_band'][bnd],
                                                                  self.models[case]['depth_band'][bnd] * u.dimensionless_unscaled,
                                                                  self.snr[band]['wave'])
                self.simdata[case][band]['wave_sampled'] = depth_spec_sampled.wavelength
                self.simdata[case][band]['depth_sampled'] = depth_spec_sampled.flux

                if case == measurement_case:

                    self.simdata[case]['measurement_case'] = True

                    # add scatter to the light curve based on the computed snr per integration
                    err_nobs_broad = self.snr[band]['err_nobs_broad']
                    scatter_nobs = np.random.normal(0, err_nobs_broad, total_int)
                    lc_nobs_scatter = lc + scatter_nobs

                    # bin up the simulated scattered light curve
                    time_binned, lc_nobs_scatter_binned, npoints_per_bin = get_binned_signal(time, lc_nobs_scatter)
                    lc_nobs_binned_error = np.sqrt(npoints_per_bin*err_nobs_broad**2) / npoints_per_bin

                    # save for later
                    self.simdata[case][band]['lc_nobs_scatter']        = lc_nobs_scatter
                    self.simdata[case][band]['total_int']              = total_int
                    self.simdata[case][band]['time_binned']            = time_binned
                    self.simdata[case][band]['lc_nobs_scatter_binned'] = lc_nobs_scatter_binned
                    self.simdata[case][band]['lc_nobs_binned_error']   = lc_nobs_binned_error
                    self.simdata[case][band]['npoints_per_bin']        = npoints_per_bin


        #for i, model in enumerate(self.models):
        #    if self.verbose: print(model)
            
        #    chisq = self.calc_chi_sq(self.models[model]['fpfs_band'], draw_eclipse_depth_measurement, yerr_nobs) / ndraws # can do this trick since only 1 data point
        #    if self.verbose: print('    chisq_red', chisq)
        #    sigma = self.calc_significance(chisq, dof)
        #    if self.verbose: print('    sigma', sigma)

        #    self.models[model]['chisq'] = chisq
        #    self.models[model]['sigma'] = sigma
            
        #    if display_figure:
        #        figure['FpFs'].plot(self.models[model]['wave'], self.models[model]['fpfs'] *1e6, lw=3, color=f'C{i%10}', label=model+f', {sigma:.2f}$\sigma$')
        #        figure['FpFs'].plot(self.models[model]['wave_band'].value, self.models[model]['fpfs_band']*1e6, 's', color=f'C{i%10}')

            # compare bare rock case to atmosphere case
       

    def read_model_spectrum(self, filepath):
        """
        Reads a model spectrum file containing wavelength and flux information.
        
        Parameters
        ----------
        filepath : str
            Path to the model spectrum file. File should be a space/tab-delimited
            file with wavelength in first column and flux in second column.
            
        Returns
        -------
        tuple
            A tuple containing (wavelength_array, flux_array) where:
            - wavelength_array is in microns
            - flux_array is in appropriate units for Fp/Fs ratio
            
        Notes
        -----
        This function is designed to read files like 'Trappist_1e_CH4_1bar_TOA_flux_eclipse.dat'
        """
        try:
            data = np.loadtxt(filepath)
            wavelength = data[:, 0]  # First column
            flux = data[:, 1]        # Second column
            return wavelength, flux
        except Exception as e:
            print(f"Error reading model file {filepath}: {str(e)}")
            return None, None

    def get_bkg(self, ref_wave, make_new_bkg=False, savepath='./background/', verbose=False):
        """
        Code to retrieve sky background from the jwst backgrounds database based on system coordinates
        JWST backgrounds: https://jwst-docs.stsci.edu/jwst-general-support/jwst-background-model
        
        Inputs:
        targ          -- dictionary of target; must include RA and Dec of system as strings
        ref_wave      -- reference wavelength for jbt to return 
        make_new_bkg  -- default=True; otherwise you can load up the last background called, 
                                       use if only working with one system
                                       
        Returns:
        background    -- list of two lists containing wavelength (um) and background counts (mJy/sr)
        """
        
        if self.verbose: print('Computing background')

        if not os.path.exists(savepath): os.makedirs(savepath)

        sys_coords = self.target['rastr']+' '+self.target['decstr']
        sys_name   = self.target['hostname'].replace(" ", "")

        def bkg():
        
            # make astropy coordinates object
            c = SkyCoord(sys_coords, unit=(u.hourangle, u.deg))

            # use jwst backgrounds to compute background at this point
            bg = jbt.background(c.ra.deg, c.dec.deg, ref_wave)

            # background is computed for many days; choose one
            ndays = bg.bkg_data['calendar'].size
            assert ndays > 0  # make sure object is visible at some point in the year; if not check coords
            middleday = bg.bkg_data['calendar'][int(ndays / 2)] # picking the middle visible date; somewhat arbitrary!
            if middleday > 365:
                print("It looks like you've picked a day outside the usual background set of 365 days.")
                print("Just going to nudge that back to 365")
                middleday = 365

            middleday_indx = np.argwhere(bg.bkg_data['calendar'] == middleday)[0][0]

            tot_bkg = bg.bkg_data['total_bg'][middleday_indx]
            wav_bkg = bg.bkg_data['wave_array']

            # background is [wavelength, total_background] in [micron, mJy/sr]
            background = [list(np.array(wav_bkg)), list(np.array(tot_bkg))]

            ascii.write(background, savepath+f"{sys_name}_background.txt", overwrite=True)
            
            if self.verbose: print('Returning background')
            return background
        
        if make_new_bkg: background = bkg()

        else: 
            try: 
                if self.verbose: print('Using existing background')
                background = ascii.read(savepath+f"{sys_name}_background.txt")
                background = [list(background['col0']), list(background['col1'])]
            except:
                background = bkg()
        
        return background

    def make_miri_dict(self, disperser, aperture, pull_default_dict=False, scene_spec='phoenix', key='m5v', input_spec=[], savepath='./'):
        """
        Code to make the initial miri dictionally for imaging_ts
        
        Inputs:
        filter            -- which photometric filter to use (e.g., f1500w)
        subarray          -- which subarray readout ot use (e.g., sub256)
        targ              -- 
        sys_coords        -- string of the coordinates of the system in RA Dec; e.g. "23h06m30.33s -05d02m36.46s";
                             to be passed to get_bkg function
        pull_default_dict -- default=True; can re-use a saved one but this doesn't save much time.
        scene_spec        -- default='phoenix' this default requires the "key" argument
                             can also be 'input' which then requires the "input_spec" argument to not be empty
        key               -- goes with scene_spec;  (e.g., 'm0v, m5v')
        input_spec        -- list of 2 arrays, or ndarray; Wavelength (μm) and Flux (mJy) arrays of the SED to use;
                             In an ndarray, wavelength is the 0th index, and flux the 1st index.
        """

        if self.verbose: print('Creating MIRI MRS dictionary')

        def pull_dict_from_stsci():
            miri_mrs_ts = build_default_calc('jwst', 'miri', 'mrs_ts', method='ifuapphot')

            # Serializing json
            json_object = json.dumps(miri_mrs_ts, indent=4)

            # Writing to sample.json
            with open("miri_mrs_ts.json", "w") as outfile:
                outfile.write(json_object)

            return miri_mrs_ts

        # grab default imaging ts dictionary (I think this only works online?)
        if pull_default_dict: miri_mrs_ts = pull_dict_from_stsci()
        else: 
            try:
                with open("miri_mrs_ts.json", "r") as f:
                    miri_mrs_ts = json.load(f)
            except(FileNotFoundError): miri_mrs_ts = pull_dict_from_stsci()

        # MIRI MRS has four channels, and three wavelength ranges. Each is calculated separately, but some are observed in parallel

        if   disperser == "short"  and aperture == "ch1": ref_wave =  5.335 * u.micron
        elif disperser == "short"  and aperture == "ch2": ref_wave =  8.154 * u.micron
        elif disperser == "short"  and aperture == "ch3": ref_wave = 12.504 * u.micron
        elif disperser == "short"  and aperture == "ch4": ref_wave = 19.290 * u.micron

        elif disperser == "medium" and aperture == "ch1": ref_wave =  6.160 * u.micron
        elif disperser == "medium" and aperture == "ch2": ref_wave =  9.415 * u.micron
        elif disperser == "medium" and aperture == "ch3": ref_wave = 14.500 * u.micron
        elif disperser == "medium" and aperture == "ch4": ref_wave = 22.485 * u.micron

        elif disperser == "long"   and aperture == "ch1": ref_wave =  7.109 * u.micron
        elif disperser == "long"   and aperture == "ch2": ref_wave = 10.850 * u.micron
        elif disperser == "long"   and aperture == "ch3": ref_wave = 16.745 * u.micron
        elif disperser == "long"   and aperture == "ch4": ref_wave = 26.220 * u.micron

        else:
            print("ERROR: I do not seem to have that disperser or aperture. Check please.")
            return {}
                
        # update with basic parameters
        miri_mrs_ts['configuration']['detector']['ngroup'] = 2
        miri_mrs_ts['configuration']['detector']['nint']   = 1
        miri_mrs_ts['configuration']['detector']['nexp']   = 1

        miri_mrs_ts['configuration']['instrument']['disperser'] = disperser # short, medium, long; defined in init
        miri_mrs_ts['configuration']['instrument']['aperture']  = aperture # e.g., ch1, ch2, ch3, ch4

        miri_mrs_ts['scene'][0]['spectrum']['normalization'] = {}
        miri_mrs_ts['scene'][0]['spectrum']['normalization']['type']          = 'photsys'
        miri_mrs_ts['scene'][0]['spectrum']['normalization']['norm_fluxunit'] = 'vegamag'
        miri_mrs_ts['scene'][0]['spectrum']['normalization']['bandpass']      = '2mass,ks'
        miri_mrs_ts['scene'][0]['spectrum']['normalization']['norm_flux']     = self.target['sy_kmag']           # change this for different stars

        if scene_spec == 'phoenix': 
            print("Using generic PHOENIX spectrum for star")
            miri_mrs_ts['scene'][0]['spectrum']['sed']['key']          = key
            miri_mrs_ts['scene'][0]['spectrum']['sed']['sed_type']     = 'phoenix'
        elif scene_spec == 'input':
            print("Using user-input spectrum for star")
            miri_mrs_ts['scene'][0]['spectrum']['sed']['sed_type']     = 'input'
            miri_mrs_ts['scene'][0]['spectrum']['sed']['spectrum']     = input_spec
            try: miri_mrs_ts['scene'][0]['spectrum']['sed'].pop('unit')
            except(KeyError): pass

        miri_mrs_ts['background'] = self.get_bkg(ref_wave, savepath=self.savepath)  
        miri_mrs_ts['background_level'] = 'high' # let's keep it conservative

        miri_mrs_ts['strategy']['reference_wavelength'] = ref_wave.value
        miri_mrs_ts['strategy']['aperture_size']        = 0.5            # default in etc
        miri_mrs_ts['strategy']['sky_annulus']          = [1.3, 1.75]    # default in etc

        if self.verbose: print('Returning MIRI dictionary')
        return miri_mrs_ts

    def make_miri_calib_dict(self, miri_dict):

        if self.verbose: print('Creating MIRI calibration dictionary')

        miri_mrs_ts_calibration = copy.deepcopy(miri_dict)

        miri_mrs_ts_calibration['scene'][0]['spectrum']['sed']['sed_type']     = 'flat'
        miri_mrs_ts_calibration['scene'][0]['spectrum']['sed']['unit']         = 'flam'
        try: miri_mrs_ts_calibration['scene'][0]['spectrum']['sed'].pop('spectrum')
        except(KeyError): pass
        try: miri_mrs_ts_calibration['scene'][0]['spectrum']['sed'].pop('key')
        except(KeyError): pass

        miri_mrs_ts_calibration['scene'][0]['spectrum']['normalization']['type']          = 'at_lambda'
        miri_mrs_ts_calibration['scene'][0]['spectrum']['normalization']['norm_wave']     = 2
        miri_mrs_ts_calibration['scene'][0]['spectrum']['normalization']['norm_waveunit'] = 'um'
        miri_mrs_ts_calibration['scene'][0]['spectrum']['normalization']['norm_flux']     = 1e-18
        miri_mrs_ts_calibration['scene'][0]['spectrum']['normalization']['norm_fluxunit'] = 'flam'
        miri_mrs_ts_calibration['scene'][0]['spectrum']['normalization'].pop('bandpass')

        if self.verbose: print('Returning MIRI calibration dictionary')

        return miri_mrs_ts_calibration

    # using batman (Kreidberg+ 2015) to make eclipse parameters
    def initialize_batman_model(self):
        
        self.batman_params = batman.TransitParams()       # object to store transit parameters
        self.batman_params.t0  = 0.0                      # time of inferior conjunction
        self.batman_params.per = 1.0                      # orbital period (phase)
        self.batman_params.rp  = self.target['pl_ratror']        # planet radius (in units of stellar radii)
        self.batman_params.a   = self.target['pl_ratdor']        # semi-major axis (in units of stellar radii)
        self.batman_params.inc = self.target['pl_orbincl']       # orbital inclination (in degrees)
        self.batman_params.ecc = 0.                       # eccentricity
        self.batman_params.w   = 90.                      # longitude of periastron (in degrees)
        self.batman_params.limb_dark = "quadratic"        # limb darkening model
        self.batman_params.u = [0.0, 0.0]                 # limb darkening coefficients [u1, u2, u3, u4]

        return self.batman_params

    def get_batman_lightcurve_tra(self, t):
        m = batman.TransitModel(self.batman_params, t)
        flux = m.light_curve(self.batman_params)
        return flux

    def get_batman_lightcurve_ecl(self, t, fp):
        self.batman_params.fp = fp
        self.batman_params.t_secondary = 0.5
        
        #t = np.linspace(0.48, 0.52, 1000)
        m = batman.TransitModel(self.batman_params, t, transittype="secondary")
        flux = m.light_curve(self.batman_params)
        return flux

    def calc_Tdur(self):
        '''
        T14 assuming eccentricity is 0
        takes: period P [days]
               radius ratio Rp_Rs []
               scaled orbital distance a_Rs []
               inclination i [deg]
        returns: duration [days]
        '''

        P     = self.target['pl_orbper']*u.day
        i     = self.target['pl_orbincl']

        if self.target['pl_ratror'] == None:
            print("Calculating Rp/Rs from separate Rp and Rs")
            Rp_Rs = ((self.target['pl_rade']*u.R_earth)/(self.target['st_rad']*u.R_sun)).decompose().value
        else: Rp_Rs = self.target['pl_ratror']
        if self.target['pl_ratdor'] == 0.0:
            print("Calculating a/Rs from separate a and Rs")
            a_Rs = ((self.target['pl_orbsmax']*u.AU)/(self.target['st_rad']*u.R_sun)).decompose().value
        else: a_Rs  = self.target['pl_ratdor']
        if self.target['pl_imppar'] == None:
            b = a_Rs * np.cos(i)
        else: b = self.target['pl_imppar']

        i = np.radians(i)

        value = np.sqrt((1 + Rp_Rs)**2 - b**2) / a_Rs / np.sin(i)

        return P/np.pi * np.arcsin(value)
        
    def calc_FpFs(self, T_p, wavelength, input_spec=[]):
        
        ''' This function will take in the Temperature in Kelvin, 
        and the wavelength range that we are looking at,
        as well as the the radius of the star and the planet. '''

        bb_p = BlackBody(T_p, scale=1*u.erg/u.s/u.cm**2/u.AA/u.sr)

        if self.target['pl_ratror'] == None:
            print("Calculating Rp/Rs from separate Rp and Rs")
            Rp_Rs = ((self.target['pl_rade']*u.R_earth)/(self.target['st_rad']*u.R_sun)).decompose().value
        else: Rp_Rs = self.target['pl_ratror']
        
        if len(input_spec) == 0:
            T_s = self.target['st_teff']*u.K
            bb_s = BlackBody(T_s, scale=1*u.erg/u.s/u.cm**2/u.AA/u.sr)

            Flux_ratio = bb_p(wavelength)/bb_s(wavelength) * (Rp_Rs)**2

        else:
            input_spec_wave = self.input_spec[0]*u.um
            input_spec_flux = self.input_spec[1]*u.mJy

            input_spec_flux *= const.c / input_spec_wave**2 / (np.pi*u.sr)
            input_spec_flux.to(u.erg/u.s/u.cm**2/u.AA/u.sr)

            interp_spec_flux = np.interp(wavelength.to(u.um), input_spec_wave, input_spec_flux)

            Flux_ratio = bb_p(wavelength) / interp_spec_flux * (Rp_Rs)**2    
            
        return Flux_ratio.decompose()

    def calc_Tday(self, albedo=0.0, atmo='bare_rock'):
        # can be 'bare rock' or 'equilibrium'
        if   atmo == 'bare_rock': f = 2/3
        elif atmo == 'equilibrium': f = 1/4
        else:
            print("ERROR: We only recognize atmos cases of bare_rock or equilibrium")
            return 0.0

        T_s = self.target['st_teff']*u.K
        if self.target['pl_ratdor'] == 0.0:
            print("Calculating a/Rs from separate a and Rs")
            a_Rs = ((self.target['pl_orbsmax']*u.AU)/(self.target['st_rad']*u.R_sun)).decompose().value
        else: a_Rs  = self.target['pl_ratdor']
        
        T_day = T_s * np.sqrt(1/a_Rs) * (1 - albedo)**(1/4) * f**(1/4)
        
        return T_day

    def inc(self, a_Rs, b):
        '''
        takes: scaled orbital distance a_Rs []
               impact paramter b []
        returns: inclination i [degrees]
        '''
        return np.degrees(np.arccos(b*(1./a_Rs)))

    def impact(self, a_Rs, i):
        '''
        takes: scaled orbital distance a_Rs []
            inclination i [degrees]
        returns: imact parameter b []
        '''
        return a_Rs * np.cos(np.radians(i))

    def get_all_model_spec_joao(targ):
        plnt_name = targ['pl_name'].replace(" ", "").replace("-", "")
        print(targ['pl_name'], '-->', plnt_name)
        
        path = '../JWST_Terrestrials/'
        specpath = path+plnt_name+'/Spectra/'
        speclist = np.sort(os.listdir(specpath))
        
        all_spec = {}
        for specfile in speclist:
            model = ascii.read(specpath+specfile, data_start=3)
        
            wave = model['col2']
            Fp_Fs = model['col7']
            all_spec[specfile] = [wave, Fp_Fs]
            
        return all_spec

    def convolve_model_spec(self, model_spec, stddev):
        
        wave, depth = model_spec[0], model_spec[1]
        
        g = Gaussian1DKernel(stddev=stddev)
        z = convolve(depth, g)
        depth_new = z

        convolved_spec = np.array([wave, depth_new])
        
        return convolved_spec

    def resample_flux_conserving(self, wave_in, flux_in, wave_out):
        """
        Resample a spectrum to a new wavelength grid while conserving total flux.
        Uses specutils FluxConservingResampler for accurate flux conservation.
        Handles NaN values by replacing them with zeros.
        
        Parameters
        ----------
        wave_in : Quantity
            Input wavelength grid (must have units)
        flux_in : Quantity
            Input flux values (must have units)
        wave_out : Quantity
            Output wavelength grid (must have units, must be convertible to wave_in units)
            
        Returns
        -------
        flux_out : Quantity
            Resampled flux values with same units as flux_in
        """
        # Require units on all inputs
        if not hasattr(wave_in, 'unit'):
            raise ValueError("wave_in must be an Astropy Quantity with units")
        if not hasattr(wave_out, 'unit'):
            raise ValueError("wave_out must be an Astropy Quantity with units")
        if not hasattr(flux_in, 'unit'):
            raise ValueError("flux_in must be an Astropy Quantity with units")
        
        # Try to convert wave_out to same units as wave_in
        try:
            wave_out = wave_out.to(wave_in.unit)
        except u.UnitConversionError:
            raise ValueError(f"Cannot convert wave_out units ({wave_out.unit}) to wave_in units ({wave_in.unit}). "
                            "Input wavelengths must have convertible units (e.g., Angstrom -> nm, but not Angstrom -> K)")
        
        # Replace NaN values with zeros in flux_in
        flux_in = flux_in.copy()
        flux_in[np.isnan(flux_in)] = 0 * flux_in.unit
            
        # Create Spectrum1D object
        spec = Spectrum1D(spectral_axis=wave_in, flux=flux_in)
        
        # Create the resampler
        resampler = FluxConservingResampler()
        
        # Create new spectral axis
        new_spec = resampler(spec, wave_out)
        
        # Replace NaN values with zeros in output
        flux_out = new_spec.flux
        flux_out[np.isnan(flux_out)] = 0 * flux_out.unit
        
        return new_spec.__class__(spectral_axis=wave_out, flux=flux_out)

    def get_bandpass_avg_flux(self, wave, depth, bandpass_wave, bandpass_filt):
        """
        Calculate a bandpass-averaged signal.
        
        """
        # Resample depth_spec onto wave_filter using flux-conserving resampling
        resampled_depth = self.resample_flux_conserving(wave, depth*u.dimensionless_unscaled, bandpass_wave).flux


        # Calculate weighted average signal over the filter
        weighted_signal = resampled_depth * bandpass_filt
        sum_weighted_signal = np.sum(weighted_signal)
        sum_filter_response = np.sum(bandpass_filt)
        average_value = sum_weighted_signal / sum_filter_response

        return average_value
        
    def grab_model_spec(targ, molecule='CO2', nbar=1):
        plnt_name = targ['pl_name'].replace(" ", "").replace("-", "")
        print(targ['pl_name'], '-->', plnt_name)
        
        path = '../JWST_Terrestrials/'
        if molecule==None: specfile = f'basaltic_noatm_TOA_flux_eclipse.dat'
        else: specfile = f'basaltic_{nbar}b_{molecule}_TOA_flux_eclipse.dat'
        specpath = path+plnt_name+'/Spectra/'+specfile
        #speclist = np.sort(os.listdir(specpath))
        
        model = ascii.read(specpath, data_start=3)
        
        wave = model['col2']
        fpfs = model['col7']
        
        return wave, fpfs
        
    def calc_chi_sq(self, model, data, error):
        return np.sum(((model - data)/error)**2)

    def calc_significance(self, chisq, dof):
        # this is a way of doing it without any 1 - *really tiny number* such that you don't get a result of inf. 
        # this website has some figures that helped me figure this out: http://work.thaslwanter.at/Stats/html/statsDistributions.html
        return stats.norm.isf((stats.chi2.sf(chisq, dof))/2.)

    def calc_APT_phase(self, obs_type='occ', Tfrac=1, Textra=0*u.hr, gamble_time=0):
        # if gamble: we will but 15 minutes of the 1 hour charge time *after* the event
        # the scheduler seems to always start the observations as early as possible so this will even out the baseline
        # however the scheduler makes no guarantees, so this is a gamle ;)
        # Tfraf =  fraction of time of out-of-event/in-event; e.g., for equal OOT to in-transit, tfrac=1
        # Textra = add some extra time to be added to the baseline in equal amounts on either side of the event
        
        # EDIT THE FOLLOWING PARAMETERS AS NECESSARY
        
        print(self.target['pl_name'])
        #tdur = orb.Tdur(P=self.target['pl_orbper']*u.day, 
        #        Rp_Rs=((self.target['pl_rade']*u.R_earth)/(self.target['st_rad']*u.R_sun)).decompose().value,
        #        a_Rs = ((self.target['pl_orbsmax']*u.AU)/(self.target['st_rad']*u.R_sun)).decompose().value,
        #        i = self.target['pl_orbincl']
        #       ) # event duration


        # if transit duration < 1 hour, automatically add a little baseline to bring it up to 1 hr; 
        # COMMMENT OUT IF YOU DO NOT WANT THIS FEATURE
        #if tdur.to(u.min).value < 60:
        #    Tfrac = ((1*u.hr)/(tdur.to(u.hr))).decompose()
        #    print(f'   padding baseline for shorter transit/ Tfrac={Tfrac}')
        #per_thresh1, ecc_thresh1 = 3, 0.05
        #per_thresh2, ecc_thresh2 = 2, 0.05
        #if self.target['pl_orbper'] > per_thresh1 and self.target['pl_orbeccen'] >= ecc_thresh1:
            # worst case scenario
        #    Tfrac = 1.2
        #    print(f'   padding baseline for orbital period >{per_thresh1} days and ecc >= {ecc_thresh1}/ Tfrac={Tfrac}')
        #elif self.target['pl_orbper'] > per_thresh2 and self.target['pl_orbeccen'] >= ecc_thresh2:
        #    Tfrac = 1.0
        #    print(f'   padding baseline for orbital period >{per_thresh2} days and ecc >= {ecc_thresh2}/ Tfrac={Tfrac}')
        #if self.target['pl_orbeccen'] >= 0.1:
        #    Tfrac = 1.5
        #    print(f'   padding baseline for large or uncertain ecc/ Tfrac={Tfrac}')
        # Dwell time of the detector in minutes, in case there is a weird ramp like with HST
        Tdwell = 30 * u.min   # 30 minutes recommended for MIRI
        # charge yourself an hour of time since you'll get charged this anyway if your window is too small (<1 hr)
        Tcharge = 1.000001 * u.hr
        # extra time you might want... perhaps to ensure you catch a secondary eclipse or whatever
        if Textra.value > 0: print(f'   Adding {Textra} to pad baseline')
        
        # T_14 in hours
        Tdur = self.tdur.to(u.hr)              # [hr]
        # Period in days
        P = self.target['pl_orbper'] * u.day
        # T0; this is a time of mid-transit when phase=0 (does not have to be for a specific transit)
        T0 = self.target['pl_tranmid']                # BJD-TDB
        # as a string, the RA and Dec of the host star, e.g., RA_Dec= '17:15:18.92 +04:57:50.1'
        RA_Dec = sys_coords = self.target['rastr']+' '+self.target['decstr']
        # do you want to observe a transit or an occultation (secondary eclsipse)

        #############################################################################################################
        ######### magical code; please don't touch ##################################################################
        #############################################################################################################

        #baseline = np.maximum((Tfrac*Tdur).to(u.hr).value, (Tdur/2).to(u.hr).value) * u.hr  # suggested pre- and post-eclipse baseline; you can tweak this if you really want
        baseline = (Tfrac*Tdur).to(u.hr) + Textra # total time out of transit/eclipse

        if   obs_type == 'tra': phase_midpoint = 0.0
        elif obs_type == 'occ': phase_midpoint = 0.5

        obs_start = Tdur/2 + baseline/2  # time before the event mid-point to start observing

        phase_max = phase_midpoint - (obs_start/P).decompose() - (Tdwell/P).decompose()
        phase_min = phase_max - (Tcharge/P).decompose()

        if gamble_time != 0: # gamble time in minutes
            print(f'   Gambling to take {gamble_time} mins before the event and stick it after (becuase observing windows seem to start on time)')
            phase_max += (gamble_time*u.min/P).decompose()
            phase_min += (gamble_time*u.min/P).decompose()

        # code shamelessly stolen from: https://gist.github.com/StuartLittlefair/4ab7bb8cf21862e250be8cb25f72bb7a
        # converts bjd_tdb to heliocentric times; need this for JWST APT
        def bary_to_helio(star, bjd, obs_name):
            bary = Time(bjd, scale='tdb', format='jd')
            obs = EarthLocation.of_site(obs_name)
            #star = SkyCoord(coords, unit=(u.hour, u.deg))
            ltt = bary.light_travel_time(star, 'barycentric', location=obs) 
            guess = bary - ltt
            delta = (guess + guess.light_travel_time(star, 'barycentric', obs)).jd  - bary.jd
            guess -= delta * u.d

            ltt = guess.light_travel_time(star, 'heliocentric', obs)
            return guess.utc + ltt

        phase0 = Time(T0, format='jd', scale='tdb')
        star = SkyCoord(RA_Dec, unit=(u.hourangle, u.deg))
        random_earth_place_name = 'lco'

        phase0_hjd = bary_to_helio(star, phase0, random_earth_place_name)

        print('   ****** PHASE *******')
        print('   Phase range {} to {}'.format(phase_min, phase_max))
        print('   Period {}'.format(P))
        print('   Zero phase (HJD) {}'.format(phase0_hjd))
        print('')


        self.total_time = (Tdwell + baseline + Tdur + Tcharge).to(u.hr)

        print('   *** ESTIMATED TOTAL TIME ***')
        print('   NOTE: This does not go into the APT, it is just an estimate')
        print('   Event duration {}'.format(Tdur.to(u.min)))
        print('   Event duration {}'.format(Tdur.to(u.hr)))
        print('   TOTAL TIME PER OBSERVATION: {}'.format(self.total_time))
        print('')
        
        
        return self.total_time

    def calc_integrations_per_exposure(self):

        if   self.subarray.lower()=='full':      tframe = 2.77504 * u.s
        elif self.subarray.lower()=='brightsky': tframe = 0.86528 * u.s
        elif self.subarray.lower()=='sub256':    tframe = 0.29952 * u.s
        elif self.subarray.lower()=='sub128':    tframe = 0.11904 * u.s
        elif self.subarray.lower()=='sub64':     tframe = 0.08500 * u.s
        else: 
            print('ERROR: subarray not recognized')
            return

        print(f'Frame time for {self.subarray} subarray: {tframe} s')
        
        tint    = tframe * self.ngroups                         # amount of time per integration
        treset  = 1*tframe                                # reset time between each integration
        cadence = tint + treset
        
        nintegrations = int(np.ceil((self.total_time/cadence).decompose()))
        print('    ***** FORM EDITOR ******')
        print(f'   subarray = {self.subarray}')
        print(f'   Ngroups {self.ngroups}')
        print(f'   Integrations/exposer {nintegrations}')
        
        return nintegrations