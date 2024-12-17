import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as u
import astropy.constants as const
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import Table
from astropy.modeling.models import BlackBody
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
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.time import Time

import warnings
warnings.filterwarnings('ignore')

print("Pandeia engine versions:", pandeia.engine.pandeia_version())

'''
This class is to help make some basic S/N calculations for observing secondary eclipse with MIRI Imaging TSO.
It shoudl keep working with updated versions of pandeia
This was adapted from a much messier code and could use some clean-up, but it's fine for now.
--- H. Diamond-Lowe 18-07-2024
'''

miri_arrays_descending = ['full', 'brightsky', 'sub256', 'sub128', 'sub64']

class MIRIImaging_Observation_Eclipse:
    def __init__(self, target, filter="f1500w", subarray="sub256", nobs=1,
                 frac_fullwell=0.65, tfrac=1,
                 find_best_subarray=False,
                 stellar_spec='default', input_spec=None,
                 verbose=True):
        self.target = target
        self.filter = filter
        self.subarray = subarray
        self.nobs = nobs
        self.frac_fullwell = frac_fullwell
        self.tfrac = tfrac
        self.find_best_subarray = find_best_subarray
        self.stellar_spec = stellar_spec
        self.input_spec = input_spec
        self.verbose = verbose

        # Initialize other attributes
        self.timing = {}
        self.miri_imaging_ts = {}
        self.miri_imaging_ts_calibration = {}


        if self.target['pl_ratror'] == None:
            print("Calculating Rp/Rs from separate Rp and Rs")
            self.target['pl_ratror'] = ((self.target['pl_rade']*u.R_earth)/(self.target['st_rad']*u.R_sun)).decompose().value
        if self.target['pl_ratdor'] == 0.0:
            print("Calculating a/Rs from separate a and Rs")
            self.target['pl_ratdor'] = ((self.target['pl_orbsmax']*u.AU)/(self.target['st_rad']*u.R_sun)).decompose().value
        if self.target['pl_orbincl'] == 0.0 or np.isnan(self.target['pl_orbincl']):
            print("Calculating inclination from impact parameter")
            self.target['pl_orbincl'] = self.inc(self.target['pl_ratdor'], self.target['pl_imppar'])

        

    def get_target_timing(self):
        '''
        stellar_spec --- 'default' --> will use pysynphot to make a PHOENIX spctrum
                         'input'   --> will use input_spec
        '''
    
        if self.verbose: 
            print(self.target)    
            print(f'***Using filter={self.filter}, subarray={self.subarray}, frac_fullwell={self.frac_fullwell}, nobs={self.nobs}')
            
        # star_params
        star_name = self.target['hostname']
        k_mag = self.target['sy_kmag']

        # need eclipse duration
        try:
            self.target['pl_ratror'].value
        except(AttributeError):
            self.target['pl_ratror'] = ((self.target['pl_rade']*u.R_earth)/(self.target['st_rad']*u.R_sun)).decompose().value

        self.tdur = self.calc_Tdur() # event duration

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
        
        if self.find_best_subarray:
            for subarray in miri_arrays_descending:     
                # make the pandeia dictionary for miri
                if self.verbose: print(f'Testing subarray {subarray}')
                miri_imaging_ts = self.make_miri_dict(subarray, scene_spec='input', input_spec=self.input_spec)

                report = perform_calculation(miri_imaging_ts)
                ngroups = int(report['scalar']['sat_ngroups']*self.frac_fullwell)
                
                self.subarray = subarray
                if ngroups >= 20: 
                    break
                    # otherwise will go with the smallest subarray
        else: 
            # make the pandeia dictionary for miri
            miri_imaging_ts = self.make_miri_dict(self.subarray, scene_spec='input', input_spec=self.input_spec)
      
            report = perform_calculation(miri_imaging_ts)
            print('ngroups before for saturation:', report['scalar']['sat_ngroups'])
            ngroups = int(report['scalar']['sat_ngroups']*self.frac_fullwell)  # use as many groups as possible without saturating, times the fraction of the full well depth
            
        if ngroups > 300: ngroups = 300        # ngroups > 300 not recommended due to cosmic rays
        elif ngroups>=5 and ngroups<=10: 
            print('ngroup is in the 5-10 range --> adding 1 group')
            ngroups+=1
        elif ngroups < mingroups:
            print('WARNING: THIS IS TOO FEW GROUPS!')
        elif ngroups < 20:
            print(f'WARNING: ngroups is {ngroups}, less than recommended ~20')
        
        self.ngroups = ngroups
        miri_imaging_ts['configuration']['detector']['ngroup'] = self.ngroups
        report = perform_calculation(miri_imaging_ts)

        if self.verbose:
            print('ETC Warnings:')
            print(report['warnings']) # should be empty if nothing is wrong
        
        tframe  = report['information']['exposure_specification']['tframe'] * u.s
        tint    = tframe * self.ngroups                         # amount of time per integration
        treset  = 1*tframe                                # reset time between each integration
        cadence = tint + treset
        nint    = (self.tdur/(tint + treset)).decompose()      # number of in-transit integrations
        ref_wave = report['scalar']['reference_wavelength']                         * u.micron
        
        if self.verbose: 
            print('Timing for each observation')
            print('    Number of groups per integration (ngroups)', ngroups)
            print('    Time to take one frame (tframe)', tframe)
            print('    Time per single integration (tframe*ngroup):', tint)
            print('    Cadence (integration time plus reset):', cadence)
            print('    Number of in-occultation integrations:', nint.decompose())
            print('    Observing efficiency (%):', (tint/cadence).decompose()*100)

        self.timing['tdur']     = self.tdur
        self.timing['nobs']     = self.nobs
        self.timing['tfrac']    = self.tfrac
        self.timing['tframe']   = tframe
        self.timing['tint']     = tint
        self.timing['treset']   = treset
        self.timing['cadence']  = cadence
        self.timing['nint']     = nint
        self.timing['ngroups']  = ngroups
        self.timing['ref_wave'] = ref_wave
        self.timing['report']   = report
        self.timing['miri_imaging_ts'] = miri_imaging_ts

        return self.timing

    def model_target_obs(self, ndraws=1000, Fp_Fs_from_model=False, convolve_model=False,
                         Albedos=[], measurement_case='bare_rock',
                         display_figure=True, save_figure=False):
        # ... (Same as the original `model_target_obs` function) ...

        Albedos = np.array(Albedos)
        assert not np.any(Albedos > 1) and not np.any(Albedos<0), "Error: cannot have albedo greater than 1 or less than 0"

        report          = self.timing['report']
        miri_imaging_ts = self.timing['miri_imaging_ts']
        tdur            = self.timing['tdur']
        tfrac           = self.timing['tfrac']
        nobs            = self.timing['nobs']
        ngroups         = self.timing['ngroups']
        cadence         = self.timing['cadence']
        ref_wave        = self.timing['ref_wave']
        nint            = self.timing['nint']


        # the wavelength and throughput of the designated filter
        bandpass_wave = report['1d']['fp'][0]
        bandpass_flux = report['1d']['fp'][1]

        # make a special dictionary, based off of the first MIRI dictionary, to get flux in useful units
        miri_imaging_ts_calibration = self.make_miri_calib_dict(miri_imaging_ts)
        report_calibration = perform_calculation(miri_imaging_ts_calibration)
        if self.verbose:
            print('Calibartion Warnings:')
            print(report_calibration['warnings']) #should be empty if nothing is wrong
        
        # compute uncertainty in a single measurement
        snr = report['scalar']['sn']
        extracted_flux = report['scalar']['extracted_flux'] / u.s
        extracted_noise = report['scalar']['extracted_noise'] / u.s

        calibration_extracted_flux = report_calibration['scalar']['extracted_flux'] / u.s
        calibration_norm_value = report_calibration['input']['scene'][0]['spectrum']['normalization']['norm_flux']

        signal = extracted_flux / calibration_extracted_flux * calibration_norm_value  * u.erg/u.s/u.cm**2/u.AA
        noise_1obs  = extracted_noise / calibration_extracted_flux * calibration_norm_value  * u.erg/u.s/u.cm**2/u.AA
        noise_nobs = noise_1obs/np.sqrt(self.nobs)

        if self.verbose: 
            print('Single integration info')
            print('    Signal +/- Noise =', signal, '+/-', noise_1obs)
            print('    Fractional uncertainty,', noise_1obs/signal)
            print('    SNR', snr)
        
            
        tstart = (self.target['pl_orbper']*u.day)*0.5 - (tdur/2) - (tdur*tfrac/2)
        tend   = (self.target['pl_orbper']*u.day)*0.5 + (tdur/2) + (tdur*tfrac/2)
        trange = tend - tstart

        total_int = int(np.ceil((trange/cadence).decompose()))

        signal_ts = np.ones(total_int)*signal
        np.random.seed(int(perf_counter()))
        
        scatter_ts_1obs = np.random.normal(0, noise_1obs.value, total_int) * u.erg/u.s/u.cm**2/u.AA
        scatter_ts_nobs = np.random.normal(0, noise_nobs.value, total_int) * u.erg/u.s/u.cm**2/u.AA

        signal_ts_scatter_1obs = signal_ts.value + scatter_ts_1obs.value
        signal_ts_scatter_nobs = signal_ts.value + scatter_ts_nobs.value
        
        # now get some model atmospehres, either from actual models, or from Teq calculations
        self.models = OrderedDict()
        
        if Fp_Fs_from_model:
            all_spec = self.get_all_model_spec(self.target, convolve_model=convolve_model)  # this is super specific to the models Joao made for the proposal 

            for model in all_spec.keys():
                wave = all_spec[model][0]
                fpfs = all_spec[model][1]
                
                bandpass_inds = (wave>bandpass_wave[0]) * (wave<bandpass_wave[-1])
                model_binned_to_bandpass = np.mean(fpfs[bandpass_inds])
                
                self.models[model] = {}
                self.models[model]['wave'] = wave
                self.models[model]['fpfs'] = fpfs
                self.models[model]['wave_band'] = ref_wave
                self.models[model]['fpfs_band'] = model_binned_to_bandpass


        wave_range = np.linspace(0.7, 25, 100) * u.micron
        for A in Albedos:
            
            # calculate the day-side temperatures in a bare rock (no circulation) model case...
            # and for an equilibrum temperature (perfect circulation) case
            for case in ['bare_rock', 'equilibrium']:
                T_day   = self.calc_Tday(A, atmo=case)
                Fp_Fs   = self.calc_FpFs(T_day, wave_range, self.input_spec)
                amp_day = self.calc_FpFs(T_day, ref_wave, self.input_spec)

                self.models[case+f'_{A}A'] = {}
                self.models[case+f'_{A}A']['wave']      = wave_range
                self.models[case+f'_{A}A']['fpfs']      = Fp_Fs
                self.models[case+f'_{A}A']['wave_band'] = ref_wave
                self.models[case+f'_{A}A']['fpfs_band'] = amp_day
                self.models[case+f'_{A}A']['T_day']     = T_day

                if self.verbose: print(case, ':', T_day)
                
        # use batman to make the system; 
        batman_params = self.initialize_batman_model()
        # create light curve time points (will use to compute fluxes)
        time = np.linspace(tstart.value, tend.value, total_int)/self.target['pl_orbper'] # time in phase
        
        def get_binned_signal(flux):
            bins = 12
            signal_ts_scatter_binned, time_bin_edges, _ = stats.binned_statistic(time, signal_ts_scatter_nobs*flux, 
                                                                                 statistic='mean', bins=bins)
            npoints_per_bin, _, _                       = stats.binned_statistic(time, signal_ts_scatter_nobs*flux, 
                                                                                 statistic='count', bins=bins)
            time_bin_width = np.mean(np.diff(time_bin_edges))
            time_binned = time_bin_edges[:-1] + time_bin_width/2
            
            return time_binned, signal_ts_scatter_binned, npoints_per_bin
        
        if display_figure:
            fig = plt.figure(figsize=(15,10))
            gs = gridspec.GridSpec(1, 3, left=0.07, right=0.99, bottom=0.1, top=0.93)

            figure = {}
            figure['lc'] = fig.add_subplot(gs[0,0:2])
            figure['FpFs'] = fig.add_subplot(gs[0,2])

        # conservative assumption for small eclipse depths -- fine to first order; DOES NOT WORK FOR TRANSITS/HJ ECLIPSES
        yerr_1obs = (1/report['scalar']['sn']) * (1/np.sqrt(nint)) * np.sqrt(1 + (1/tfrac))
        yerr_nobs = yerr_1obs * (1/np.sqrt(nobs))
        self.yerr_1obs = yerr_1obs
        self.yerr_nobs = yerr_nobs

        #if Fp_Fs_from_model:
        #    # based on Joao's model where the bare rock case is called "noatm"
        #    bare_rock_key = [x for x in self.models.keys() if 'alb01' in x]
        #    bar_atm_key   = [x for x in self.models.keys() if '10b' in x]
        #    
        #    cases = np.hstack([bare_rock_key, bar_atm_key])

        #else: cases = np.hstack(list(self.models.keys()))
        cases = np.hstack(list(self.models.keys()))
            
        line_styles = ['-', '--', ':']

        np.random.seed(50)
        if Fp_Fs_from_model==False: measurement_case = f'{measurement_case}_{Albedos[0]}A'
        
        draw_eclipse_depth_measurement = np.random.normal(self.models[measurement_case]['fpfs_band'], yerr_nobs, ndraws)

        dof = 1  # rough estimate; only 1 data point
        
        for i, case in enumerate(cases):       
            flux = self.get_batman_lightcurve(self.models[case]['fpfs_band'], time)
            time_binned, signal_ts_scatter_binned, npoints_per_bin = get_binned_signal(flux)
            
            if case==measurement_case: 

                binned_error = np.sqrt(npoints_per_bin*noise_nobs**2) / npoints_per_bin / signal

                if display_figure:

                    for n in range(self.nobs):
                        np.random.seed(int(perf_counter())+n)
        
                        scatter_ts_1obs = np.random.normal(0, noise_1obs.value, total_int) * u.erg/u.s/u.cm**2/u.AA
                        signal_ts_scatter_1obs = signal_ts.value + scatter_ts_1obs.value

                        if n==0: label = f'Cadence={np.round(cadence, 2)}; ngroups={ngroups}'
                        else: label = None
                        figure['lc'].plot(time, ((signal_ts_scatter_1obs*flux/signal).value-1)*1e6 , 
                                          '.', color='k', alpha=0.3, markeredgecolor='w',
                                          label=label)
                    
                    figure['lc'].errorbar(time_binned, ((signal_ts_scatter_binned/signal).value -1)*1e6, 
                                      yerr=binned_error*1e6, fmt='o', color='k', alpha=1)
                    
                    # only plot error bar for bare rock case in Fp/Fs figure
                    figure['FpFs'].errorbar(self.models[case]['wave_band'].value, self.models[case]['fpfs_band'] *1e6, yerr=yerr_nobs.value *1e6, fmt='.', color='k', alpha=0.8, zorder=1000)
                if self.verbose: print('Data point:', self.models[case]['fpfs_band']*1e6, '+/-', yerr_nobs*1e6, 'ppm')

            if display_figure: figure['lc'].plot(time, ((signal_ts*flux/signal).value-1)*1e6, 
                                                ls=line_styles[i%3], lw=3, color=f'C{i%10}', label=case)
            
        for i, model in enumerate(self.models):
            if self.verbose: print(model)
            
            chisq = self.calc_chi_sq(self.models[model]['fpfs_band'], draw_eclipse_depth_measurement, yerr_nobs) / ndraws # can do this trick since only 1 data point
            if self.verbose: print('    chisq_red', chisq)
            sigma = self.calc_significance(chisq, dof)
            if self.verbose: print('    sigma', sigma)

            self.models[model]['chisq'] = chisq
            self.models[model]['sigma'] = sigma
            
            if display_figure:
                figure['FpFs'].plot(self.models[model]['wave'], self.models[model]['fpfs'] *1e6, lw=3, color=f'C{i%10}', label=model+f', {sigma:.2f}$\sigma$')
                figure['FpFs'].plot(self.models[model]['wave_band'].value, self.models[model]['fpfs_band']*1e6, 's', color=f'C{i%10}')

            # compare bare rock case to atmosphere case
        
        if display_figure:
            figure['lc'].axvline(0.5, ls=':', color='k', alpha=0.5)
            figure['lc'].axvline(0.5-tdur.value/self.target['pl_orbper']/2, ls='--', color='k', alpha=0.5)
            figure['lc'].axvline(0.5+tdur.value/self.target['pl_orbper']/2, ls='--', color='k', alpha=0.5)

            figure['lc'].legend(loc='upper right')
            per = self.target['pl_orbper']
            k_mag = self.target['sy_kmag']
            figure['lc'].set_title(self.target['pl_name']+f', Kmag={k_mag}, {nobs} obs, Tdur = {np.round(tdur.to(u.min), 2)}, P={np.round(per, 3)} days', fontsize=16)

            figure['lc'].set_xlabel('Phase', fontsize=14)
            figure['lc'].set_ylabel('Normalized Flux (ppm)', fontsize=14)

            figure['lc'].grid(alpha=0.4)
            figure['lc'].set_ylim(-0.4*np.std(((signal_ts_scatter_1obs*flux/signal).value-1)*1e6), 1.5*np.std((signal_ts_scatter_1obs*flux/signal).value-1)*1e6)
            
            figure['FpFs'].legend(loc='upper left')
            figure['FpFs'].set_ylabel('$F_p$/$F_s$ (ppm)', fontsize=14)
            figure['FpFs'].set_xlabel('Wavelength ($\mu$m)', fontsize=14)
        
            if Fp_Fs_from_model: figure['FpFs'].set_title(f'{nobs} obs', fontsize=16)
            else:
                T_rock = np.rint(self.models[f'bare_rock_{Albedos[0]}A']['T_day'])
                figure['FpFs'].set_title(f'T_day,rock = {T_rock}, {nobs} obs', fontsize=16)

            figure['FpFs'].set_xlim(0.7, 25)
            figure['FpFs'].grid(alpha=0.4)
            #figure['FpFs'].set_ylim(-10, 400)

        plname = self.target['pl_name'].replace(' ','')  # w/o spaces
        if save_figure: plt.savefig(f'../sample/model_observations/{plname}_{filter}_{subarray}_{nobs}obs.png', facecolor='white')
        if display_figure: plt.show()
        #plt.close()
        
        return

    def get_bkg(self, ref_wave, make_new_bkg=True, savepath='./background/', verbose=False):
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

    def make_miri_dict(self, subarray, pull_default_dict=False, scene_spec='phoenix', key='m5v', input_spec=[], savepath='./'):
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

        if self.verbose: print('Creating MIRI dictionary')

        def pull_dict_from_stsci():
            miri_imaging_ts = build_default_calc('jwst', 'miri', 'imaging_ts')

            # Serializing json
            json_object = json.dumps(miri_imaging_ts, indent=4)

            # Writing to sample.json
            with open("miri_imaging_ts.json", "w") as outfile:
                outfile.write(json_object)

            return miri_imaging_ts

        # grab default imaging ts dictionary (I think this only works online?)
        if pull_default_dict: miri_imaging_ts = pull_dict_from_stsci()
        else: 
            try:
                with open("miri_imaging_ts.json", "r") as f:
                    miri_imaging_ts = json.load(f)
            except(FileNotFoundError): miri_imaging_ts = pull_dict_from_stsci()
                
        if   self.filter == 'f1500w': ref_wave = 15.0 * u.micron
        elif self.filter == 'f1800w': ref_wave = 18.0 * u.micron
        elif self.filter == 'f1280w': ref_wave = 12.8 * u.micron
        elif self.filter == 'f2100w': ref_wave = 21.0 * u.micron
        elif self.filter == 'f2550w': ref_wave = 25.5 * u.micron
        elif self.filter == 'f1130w': ref_wave = 11.3 * u.micron
        elif self.filter == 'f1000w': ref_wave = 10.0 * u.micron
        elif self.filter == 'f770w' : ref_wave =  7.7 * u.micron
        elif self.filter == 'f560w' : ref_wave =  5.6 * u.micron
        else:
            print("ERROR: I do not seem to have that filter. Check please.")
            return {}
                
        # update with basic parameters
        miri_imaging_ts['configuration']['instrument']['filter'] = self.filter
        miri_imaging_ts['configuration']['detector']['subarray'] = subarray
           
        miri_imaging_ts['configuration']['detector']['ngroup'] = 2    
        miri_imaging_ts['configuration']['detector']['nint']   = 1 
        miri_imaging_ts['configuration']['detector']['nexp']   = 1
        miri_imaging_ts['configuration']['detector']['readout_pattern'] = 'fastr1'
        #try: miri_imaging_ts['configuration'].pop('max_filter_leak')
        #except(KeyError): pass

        miri_imaging_ts['scene'][0]['spectrum']['normalization'] = {}
        miri_imaging_ts['scene'][0]['spectrum']['normalization']['type']          = 'photsys'
        miri_imaging_ts['scene'][0]['spectrum']['normalization']['norm_fluxunit'] = 'vegamag'
        miri_imaging_ts['scene'][0]['spectrum']['normalization']['bandpass']      = '2mass,ks'
        miri_imaging_ts['scene'][0]['spectrum']['normalization']['norm_flux']     = self.target['sy_kmag']           # change this for different stars

        if scene_spec == 'phoenix': 
            miri_imaging_ts['scene'][0]['spectrum']['sed']['key']          = key
            miri_imaging_ts['scene'][0]['spectrum']['sed']['sed_type']     = 'phoenix'
            #try: miri_imaging_ts['scene'][0]['spectrum']['sed'].pop('unit')
            #except(KeyError): pass
        elif scene_spec == 'input':
            miri_imaging_ts['scene'][0]['spectrum']['sed']['sed_type']     = 'input'
            miri_imaging_ts['scene'][0]['spectrum']['sed']['spectrum']     = input_spec
            try: miri_imaging_ts['scene'][0]['spectrum']['sed'].pop('unit')
            except(KeyError): pass
            #try: miri_imaging_ts['scene'][0]['spectrum']['sed'].pop('key')
            #except(KeyError): pass

        miri_imaging_ts['background'] = self.get_bkg(ref_wave)
        miri_imaging_ts['background_level'] = 'high' # let's keep it conservative

        miri_imaging_ts['strategy']['aperture_size']  = 0.55            # values from Greene+ 2023 for T1-b
        miri_imaging_ts['strategy']['sky_annulus']    = [1.32, 2.8]     # assuming MIRI plate scale of 0.11"/pix

        if self.verbose: print('Returning MIRI dictionary')
        return miri_imaging_ts

    def make_miri_calib_dict(self, miri_dict):

        if self.verbose: print('Creating MIRI calibration dictionary')

        miri_imaging_ts_calibration = copy.deepcopy(miri_dict)

        miri_imaging_ts_calibration['scene'][0]['spectrum']['sed']['sed_type']     = 'flat'
        miri_imaging_ts_calibration['scene'][0]['spectrum']['sed']['unit']         = 'flam'
        try: miri_imaging_ts_calibration['scene'][0]['spectrum']['sed'].pop('spectrum')
        except(KeyError): pass
        try: miri_imaging_ts_calibration['scene'][0]['spectrum']['sed'].pop('key')
        except(KeyError): pass

        miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization']['type']          = 'at_lambda'
        miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization']['norm_wave']     = 2
        miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization']['norm_waveunit'] = 'um'
        miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization']['norm_flux']     = 1e-18
        miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization']['norm_fluxunit'] = 'flam'
        miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization'].pop('bandpass')

        if self.verbose: print('Returning MIRI calibration dictionary')

        return miri_imaging_ts_calibration

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
        self.batman_params.u = [0.5, 0.1]                 # limb darkening coefficients [u1, u2, u3, u4]

        return self.batman_params

    def get_batman_lightcurve(self, fp, t):
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

        i = np.radians(i)
        b = a_Rs * np.cos(i)

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

    def get_all_model_spec(self, targ, convolve_model=False):
        plnt_name = targ['pl_name'].replace(" ", "").replace("-", "")
        print(targ['pl_name'], '-->', plnt_name)
        
        modelspecpath = f'./models/{plnt_name.lower()}/'
        speclist = np.sort(os.listdir(modelspecpath))
        
        all_spec = {}
        for specfile in speclist:
            model = ascii.read(modelspecpath + specfile, data_start=3)
        
            wave = model['col2']
            Fp_Fs = model[model.colnames[-1]]

            if convolve_model!=False:
                g = Gaussian1DKernel(stddev=convolve_model)
                z = convolve(Fp_Fs, g)
                Fp_Fs = z

            spec_name = specfile.split(".")[0]

            all_spec[spec_name] = [wave, Fp_Fs]
            
        return all_spec
        
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