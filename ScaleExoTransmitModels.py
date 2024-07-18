import numpy as np
import matplotlib.pyplot as plt
import astrotools.modeldepths as md
import astropy.units as u
import astropy.constants as c
from astrotools.species_dict import *
import sys
from astropy.convolution import convolve, Gaussian1DKernel

class ScaleExoTransmitModels(object):
    ''' 
    '''

    def __init__(self, targ_name, targ_Rs, targ_Rp, targ_T, targ_g, species, extranameold, tracer, numH, bin_size, wave_range, model_T, model_g, smooth_model, model_code):

        self.model_code = model_code
        if model_code == 'ExoTransmit': print('     [scale exotransmit models] initializing scaling object')
        elif model_code == 'GoldenRetriever': print(     '[scale goldie models] initializing scaling object')
        else: print('ERROR: You must use one of the pre-defined modeling codes!')

        self.targ_name = targ_name
        self.targ_Rs = targ_Rs      # [solar radii]
        self.targ_Rp = targ_Rp       # [earth radii]
        self.targ_T = targ_T        # [K]
        self.targ_g = targ_g        # [m/s^2]

        self.species_dict = create_species_dict()      # dictionary of species and their corresponding mean molecular weights
        self.species = species            # list of molecular species you are interested in
        self.extranameold = extranameold
        self.tracer = tracer
        #self.numH = numH               # integer number of scale heights you estimate you can probe
        self.bin_size = bin_size           # [microns]
        self.wave_range = wave_range         # [microns]

        self.model_T = model_T
        self.model_g = model_g
        self.smooth_model = smooth_model


    # scale height calculation
    def scale_height(self, T, mu, g):
        # takes effective temperature [K], mu is weighted average of molecules, g is surface gravity in m/s^2
        mH = c.m_p + c.m_e	    # mass of hydrogen atom [kg]
        print((c.k_B*T/(mu*mH*g)).decompose())
        return (c.k_B*T/(mu*mH*g)).decompose()

    # change in transit depth
    def Delta_transit_depth(self, H, Rp, Rs):       
        # takes scale height [km], planet radius [km], stellar radius [km]
        print('Delta_transit_depth', (((Rp + H)/Rs)**2 - (Rp/Rs)**2).decompose())
        return (((Rp + H)/Rs)**2 - (Rp/Rs)**2).decompose()

    def make_model_depths(self):

        self.model_dict = {}
        for s in self.species:
            self.model_dict[s] = {}
            # retrieve a set of wavelengths, depths, and bin-averaged wavelengths and depths from the ExoTransmit models
            if self.model_code == 'ExoTransmit':
                modelfile = self.targ_name+'_'+s+self.extranameold+'.dat'
                try: 
                    print('     [scale exotransmit models] trying to load model from {0}'.format(modelfile))

                    model = md.model_spectrum(modelfile, self.bin_size.value, self.wave_range.value, smooth=self.smooth_model, model_code=self.model_code)
                except(IOError): 
                    sys.exit( '     [scale exotransmit models] you need to make an ExoTransmit model named {0} first!'.format(modelfile))
            elif self.model_code == 'GoldenRetriever':
                modelfile = 'Transmission'+'_'+s+'_'+self.extranameold+'_'+self.targ_name+'.h5'
                try: 
                    print('     [scale goldie models] trying to load model from {0}'.format(modelfile))

                    model = md.model_spectrum(modelfile, self.bin_size.value, self.wave_range.value, smooth=self.smooth_model, model_code=self.model_code)
                except(IOError): 
                    sys.exit( '     [scale goldie models] you need to make a GoldenRetriever model named {0} first!'.format(modelfile))                
                

            self.model_dict[s]['wavelengths'] = model[0] * u.micron
            self.model_dict[s]['depths'] = model[1] 
            self.model_dict[s]['avg_wavelengths'] = model[2] * u.micron
            self.model_dict[s]['avg_depths'] = model[3]

            # get the scale height and depth that comes directly from the parameters used to make the ExoTransmit model
            if s == 'flat':
                scaleheight_model = self.scale_height(self.model_T, self.species_dict[self.species[0]], self.model_g)
                depth_model = self.Delta_transit_depth(scaleheight_model, self.targ_Rp, self.targ_Rs)
            else:
                scaleheight_model = self.scale_height(self.model_T, self.species_dict[s], self.model_g)
                depth_model = self.Delta_transit_depth(scaleheight_model, self.targ_Rp, self.targ_Rs)

            # scale the model to the more accurate parameters of the planet
            scaleheight_species = self.scale_height(self.targ_T, self.species_dict[s], self.targ_g)
            depth_species = self.Delta_transit_depth(scaleheight_species, self.targ_Rp, self.targ_Rs)
            scale = (depth_species/depth_model).decompose()
            print('     [scale exotransmit models] scale for species {0} is {1}'.format(s, scale))
            print(scaleheight_model, depth_model, scaleheight_species, depth_species)


            # the depth around which the transmisison wiggles are placed
            refmean = np.mean(self.model_dict[self.species[0]]['depths'])

            # make Delta_depths; a squiggly line aroun 0 that can be scaled and added back to refmean
            model_depths = np.array(self.model_dict[s]['depths'])
            mean_model_depths = np.mean(model_depths)
            Delta_depths = model_depths - mean_model_depths
            self.model_dict[s]['model_scaled'] = refmean + Delta_depths*scale

            model_depths_binned = np.array(self.model_dict[s]['avg_depths'])
            mean_model_depths_binned = np.mean(model_depths_binned)
            Delta_depths_binned = model_depths_binned - mean_model_depths_binned
            self.model_dict[s]['model_scaled_binned'] = refmean + Delta_depths_binned*scale

        return self.model_dict

