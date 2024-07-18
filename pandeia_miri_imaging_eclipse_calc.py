import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as u
import astropy.constants as const
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
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

print("Pandeia engine versions:", pandeia.engine.pandeia_version())

miri_arrays_descending = ['full', 'brightsky', 'sub256', 'sub128', 'sub64']

#from time import time_ns

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

    def get_target_timing(self, verbose=True):
        '''
        stellar_spec --- 'default' --> will use pysynphot to make a PHOENIX spctrum
                         'input'   --> will use input_spec
        '''
    
        if verbose: 
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
        if verbose: print('Approximate observing time per eclipse observation:', approx_obs_time)

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
                print(f'Testing subarray {subarray}')
                miri_imaging_ts = self.make_miri_dict(subarray, scene_spec='input', input_spec=self.input_spec, verbose=verbose)

                report = perform_calculation(miri_imaging_ts)
                ngroups = int(report['scalar']['sat_ngroups']*self.frac_fullwell)
                
                if ngroups >= 20: 
                    self.subarray = subarray
                    break
        else: 
            # make the pandeia dictionary for miri
            miri_imaging_ts = self.make_miri_dict(self.subarray, scene_spec='input', input_spec=self.input_spec, verbose=verbose)
      
            report = perform_calculation(miri_imaging_ts)
            print('ngroups before for saturation:', report['scalar']['sat_ngroups'])
            ngroups = int(report['scalar']['sat_ngroups']*self.frac_fullwell)  # use as many groups as possible without saturating, times the fraction of the full well depth
            
        if ngroups > 300: ngroups = 300        # ngroups > 300 not recommended due to cosmic rays
        elif ngroups>=5 and ngroups<=10: 
            print('ngroup is in the 5-10 range --> adding 1 group')
            ngroups+=1
        elif ngroups < mingroups:
            print('WARNING: THIS IS TOO FEW GROUPS!')
        
        self.ngroups = ngroups
        miri_imaging_ts['configuration']['detector']['ngroup'] = self.ngroups
        report = perform_calculation(miri_imaging_ts)

        print('ETC Warnings:')
        print(report['warnings']) # should be empty if nothing is wrong
        
        tframe  = report['information']['exposure_specification']['tframe'] * u.s
        tint    = tframe * self.ngroups                         # amount of time per integration
        treset  = 1*tframe                                # reset time between each integration
        cadence = tint + treset
        nint    = (self.tdur/(tint + treset)).decompose()      # number of in-transit integrations
        ref_wave = report['scalar']['reference_wavelength']                         * u.micron
        
        if verbose: 
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

    def model_target_obs(self, ndraws=1000, Fp_Fs_from_model=False, Albedos=[],
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
        print('Calibartion Warnings:')
        print(report_calibration['warnings']) #should be empty if nothing is wrong
        
        # compute uncertainty in a single measurement
        snr = report['scalar']['sn']
        extracted_flux = report['scalar']['extracted_flux'] / u.s
        extracted_noise = report['scalar']['extracted_noise'] / u.s

        calibration_extracted_flux = report_calibration['scalar']['extracted_flux'] / u.s
        calibration_norm_value = report_calibration['input']['scene'][0]['spectrum']['normalization']['norm_flux']

        signal = extracted_flux / calibration_extracted_flux * calibration_norm_value  * u.erg/u.s/u.cm**2/u.AA
        noise  = extracted_noise / calibration_extracted_flux * calibration_norm_value  * u.erg/u.s/u.cm**2/u.AA

        if self.verbose: 
            print('Single integration info')
            print('    Signal +/- Noise =', signal, '+/-', noise)
            print('    Fractional uncertainty,', noise/signal)
            print('    SNR', snr)
        
        noise /= np.sqrt(self.nobs)
            
        tstart = (self.target['pl_orbper']*u.day)*0.5 - (tdur/2) - (tdur*tfrac/2)
        tend   = (self.target['pl_orbper']*u.day)*0.5 + (tdur/2) + (tdur*tfrac/2)
        trange = tend - tstart
        total_int = int(np.ceil((trange/cadence).decompose()))

        signal_ts = np.ones(total_int)*signal
        np.random.seed(int(perf_counter()))
        scatter_ts = np.random.normal(0, noise.value, total_int) * u.erg/u.s/u.cm**2/u.AA
        signal_ts_scatter = signal_ts.value + scatter_ts.value
        
        # now get some model atmospehres, either from actual models, or from Teq calculations
        models = OrderedDict()
        
        if Fp_Fs_from_model:
            all_spec = get_all_model_spec(self.target)  # this is super specific to the models Joao made for the proposal 
            for model in all_spec.keys():
                wave = all_spec[model][0]
                fpfs = all_spec[model][1]
                
                bandpass_inds = (wave>bandpass_wave[0]) * (wave<bandpass_wave[-1])
                model_binned_to_bandpass = np.mean(fpfs[bandpass_inds])
                
                models[model] = {}
                models[model]['wave'] = wave
                models[model]['fpfs'] = fpfs
                models[model]['wave_band'] = ref_wave
                models[model]['fpfs_band'] = model_binned_to_bandpass


        wave_range = np.linspace(0.7, 25, 100) * u.micron
        for A in Albedos:
            
            # calculate the day-side temperatures in a bare rock (no circulation) model case...
            # and for an equilibrum temperature (perfect circulation) case
            for case in ['bare_rock', 'equilibrium']:
                T_day   = self.calc_Tday(A, atmo=case)
                Fp_Fs   = self.calc_FpFs(T_day, wave_range)
                amp_day = self.calc_FpFs(T_day, ref_wave)

                models[case+f'_{A}A'] = {}
                models[case+f'_{A}A']['wave']      = wave_range
                models[case+f'_{A}A']['fpfs']      = Fp_Fs
                models[case+f'_{A}A']['wave_band'] = ref_wave
                models[case+f'_{A}A']['fpfs_band'] = amp_day
                models[case+f'_{A}A']['T_day']     = T_day

                print(case, ':', T_day)
                
        # use batman to make the system; 
        batman_params = self.initialize_batman_model()
        # create light curve time points (will use to compute fluxes)
        time = np.linspace(tstart.value, tend.value, total_int)/self.target['pl_orbper'] # time in phase
        
        def get_binned_signal(flux):
            bins = 12
            signal_ts_scatter_binned, time_bin_edges, _ = stats.binned_statistic(time, signal_ts_scatter*flux, 
                                                                                 statistic='mean', bins=bins)
            npoints_per_bin, _, _                       = stats.binned_statistic(time, signal_ts_scatter*flux, 
                                                                                 statistic='count', bins=bins)
            time_bin_width = np.mean(np.diff(time_bin_edges))
            time_binned = time_bin_edges[:-1] + time_bin_width/2
            
            return time_binned, signal_ts_scatter_binned, npoints_per_bin
        
        fig = plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(1, 3, left=0.07, right=0.99, bottom=0.1, top=0.93)

        figure = {}
        figure['lc'] = fig.add_subplot(gs[0,0:2])
        figure['FpFs'] = fig.add_subplot(gs[0,2])

        # conservative assumption for small eclipse depths -- fine to first order
        yerr = (1/report['scalar']['sn']) * (1/np.sqrt(nint)) * (1/np.sqrt(nobs)) * np.sqrt(1 + (1/tfrac))

        if Fp_Fs_from_model:
            # based on Joao's model where the bare rock case is called "noatm"
            bare_rock_key = [x for x in models.keys() if 'alb01' in x]
            bar_atm_key   = [x for x in models.keys() if '10b' in x]
            
            cases = np.hstack([bare_rock_key, bar_atm_key])

        else: cases = np.hstack(list(models.keys()))
            
        line_styles = ['-', '--', ':']
        colors = ['C3', 'C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

        np.random.seed(50)
        draw_eclipse_depth_bare_rock = np.random.normal(models[cases[0]]['fpfs_band'], yerr, ndraws)
        dof = 1  # rough estimate; only 1 data point
        
        for i, case in enumerate(cases):       
            flux = self.get_batman_lightcurve(models[case]['fpfs_band'], time)
            time_binned, signal_ts_scatter_binned, npoints_per_bin = get_binned_signal(flux)
            
            if i==0: 
                figure['lc'].plot(time, ((signal_ts_scatter*flux/signal).value-1)*1e6 , 
                                  '.', color='k', alpha=0.3, markeredgecolor='w',
                                  label=f'Cadence={np.round(cadence, 2)}; ngroups={ngroups}')
                
                binned_error = np.sqrt(npoints_per_bin*noise**2) / npoints_per_bin / signal
                figure['lc'].errorbar(time_binned, ((signal_ts_scatter_binned/signal).value -1)*1e6, 
                                  yerr=binned_error*1e6, fmt='o', color='k', alpha=1)
                
                # only plot error bar for bare rock case in Fp/Fs figure
                figure['FpFs'].errorbar(models[case]['wave_band'].value, models[case]['fpfs_band'] *1e6, yerr=yerr.value *1e6, fmt='.', color='k', alpha=0.8, zorder=1000)
                print('Data point:', models[case]['fpfs_band']*1e6, '+/-', yerr*1e6, 'ppm')

            figure['lc'].plot(time, ((signal_ts*flux/signal).value-1)*1e6, 
                              ls=line_styles[i%3], lw=3, color=colors[i], label=case)
            
        for i, model in enumerate(models):
            print(model)
            
            chisq = self.calc_chi_sq(models[model]['fpfs_band'], draw_eclipse_depth_bare_rock, yerr) / ndraws # can do this trick since only 1 data point
            print('    chisq_red', chisq)
            sigma = self.calc_significance(chisq, dof)
            print('    sigma', sigma)
            
            figure['FpFs'].plot(models[model]['wave'], models[model]['fpfs'] *1e6, lw=3, color=colors[i%10], label=model+f', {sigma:.2f}$\sigma$')
            figure['FpFs'].plot(models[model]['wave_band'].value, models[model]['fpfs_band']*1e6, 's', color=colors[i%10])

            # compare bare rock case to atmosphere case
        
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
        figure['lc'].set_ylim(-0.3*np.std(((signal_ts_scatter*flux/signal).value-1)*1e6), 0.3*np.std((signal_ts_scatter*flux/signal).value-1)*1e6)
        
        figure['FpFs'].legend(loc='upper left')
        figure['FpFs'].set_ylabel('$F_p$/$F_s$ (ppm)', fontsize=14)
        figure['FpFs'].set_xlabel('Wavelength ($\mu$m)', fontsize=14)
        
        if Fp_Fs_from_model: figure['FpFs'].set_title(f'{nobs} obs', fontsize=16)
        else:
            T_rock = np.rint(models[f'bare_rock_{Albedos[0]}A']['T_day'])
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
        
        print('Computing background')

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
            
            print('Returning background')
            return background
        
        if make_new_bkg: background = bkg()

        else: 
            try: 
                print('Using existing background')
                background = ascii.read(savepath+f"{sys_name}_background.txt")
                background = [list(background['col0']), list(background['col1'])]
            except:
                background = bkg()
        
        return background

    def make_miri_dict(self, subarray, pull_default_dict=False, scene_spec='phoenix', key='m5v', input_spec=[], savepath='./', verbose=False):
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

        if verbose: print('Creating MIRI dictionary')

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

        miri_imaging_ts['background'] = self.get_bkg(ref_wave, verbose=verbose)
        miri_imaging_ts['background_level'] = 'high' # let's keep it conservative

        miri_imaging_ts['strategy']['aperture_size']  = 0.55            # values from Greene+ 2023 for T1-b
        miri_imaging_ts['strategy']['sky_annulus']    = [1.32, 2.8]     # assuming MIRI plate scale of 0.11"/pix

        if verbose: print('Returning MIRI dictionary')
        return miri_imaging_ts

    def make_miri_calib_dict(self, miri_dict):

        print('Creating MIRI calibration dictionary')

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

        print('Returning MIRI calibration dictionary')

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
        Rp_Rs = self.target['pl_ratror']
        a_Rs  = self.target['pl_ratdor']
        i     = self.target['pl_orbincl']

        i = np.radians(i)
        b = a_Rs * np.cos(i)

        value = np.sqrt((1 + Rp_Rs)**2 - b**2) / a_Rs / np.sin(i)
        return P/np.pi * np.arcsin(value)
        
    def calc_FpFs(self, T_p, wavelength):
        
        ''' This function will take in the Temperature in Kelvin, 
        and the wavelength range that we are looking at,
        as well as the the radius of the star and the planet. '''

        T_s = self.target['st_teff']*u.K
        Rp_Rs = self.target['pl_ratror']
        
        bb_s = BlackBody(T_s, scale=1*u.erg/u.s/u.cm**2/u.AA/u.sr)
        bb_p = BlackBody(T_p, scale=1*u.erg/u.s/u.cm**2/u.AA/u.sr)
        
        Flux_ratio = bb_p(wavelength)/bb_s(wavelength) * (Rp_Rs)**2
            
        return Flux_ratio.decompose()

    def calc_Tday(self, albedo=0.0, atmo='bare_rock'):
        # can be 'bare rock' or 'equilibrium'
        if   atmo == 'bare_rock': f = 2/3
        elif atmo == 'equilibrium': f = 1/4

        T_s = self.target['st_teff']*u.K
        a_Rs = self.target['pl_ratdor']
        
        T_day = T_s * np.sqrt(1/a_Rs) * (1 - albedo)**(1/4) * f**(1/4)
        
        return T_day

    def get_all_model_spec(targ):
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