import numpy as np
import matplotlib.pyplot as plt
import astrotools.modeldepths as md
import astropy.units as u
import astropy.constants as c
import random
import scipy.stats as st


class AtmospheresUVIS(object):
    ''' 
    '''

    def __init__(self, targ_name, targ_Rs, targ_Rp, targ_T, targ_g, species_dict, species, tracer, numH, bin_size, wave_range, model_T, model_g, smooth_model, wave_centers, phot_rms):
        print('     [atmospheres] initializing atmospheres object')
        self.targ_name = targ_name
        self.targ_Rs = targ_Rs      # [solar radii]
        self.targ_Rp = targ_Rp       # [earth radii]
        self.targ_T = targ_T        # [K]
        self.targ_g = targ_g        # [m/s^2]

        self.species_dict = species_dict       # dictionary of species and their corresponding mean molecular weights
        self.species = species            # list of molecular species you are interested in
        self.tracer = tracer
        self.numH = numH               # integer number of scale heights you estimate you can probe
        self.bin_size = bin_size           # [microns]
        self.wave_range = wave_range         # [microns]

        self.model_T = model_T
        self.model_g = model_g
        self.smooth_model = smooth_model

        self.wave_centers = wave_centers
        self.phot_rms = phot_rms

        # !! need to account for duration, stellar size, and planet size between target and reference

    # scale height calculation
    def scale_height(self, T, mu, g):
        # takes effective temperature [K], mu is weighted average of molecules, g is surface gravity in m/s^2
        mH = c.m_p + c.m_e	    # mass of hydrogen atom [kg]
        return (c.k_B*T/(mu*mH*g)).decompose()

    def Delta_transit_depth(self, H, numH, Rp, Rs):       
        # takes scale height [km], number of scale heights, planet radius [km], stellar radius [km]
        return (((Rp + numH*H)/Rs)**2 - (Rp/Rs)**2).decompose()

    def make_model_depths(self):

        self.model_dict = {}
        for s in self.species:
            self.model_dict[s] = {}
            modelfile = self.targ_name+'_'+s+'.dat'
            # retrieve a set of wavelengths, depths, and bin-averaged wavelengths and depths from the ExoTransmit models
            try: 
                print('     [atmospheres] trying to load model from {0}'.format(modelfile))
                model = md.model_spectrum(modelfile, self.bin_size.value, self.wave_range.value, smooth=self.smooth_model)
            except(IOError): 
                print('     [atmospheres] you need to make an ExoTransmit model named {0} first!'.format(modelfile))
                return

            self.model_dict[s]['wavelengths'] = model[0] * u.micron
            self.model_dict[s]['depths'] = model[1] 
            self.model_dict[s]['avg_wavelengths'] = model[2] * u.micron
            self.model_dict[s]['avg_depths'] = model[3]

            scaleheight_model = self.scale_height(self.model_T, self.species_dict[self.species[0]], self.model_g)
            depth_model = self.Delta_transit_depth(scaleheight_model, 1, self.targ_Rp, self.targ_Rs)

            if s == 'Earth':
                self.model_g = 9.81 * (u.m/u.s**2)
                self.model_T = 280 * u.K
                model_Rs = 1. * u.R_sun
                model_Rp = 1. * u.R_earth
                scaleheight_model = self.scale_height(self.model_T, self.species_dict[self.species[0]], self.model_g)
                depth_model = self.Delta_transit_depth(scaleheight_model, 1, model_Rp, model_Rs)

            scaleheight_species = self.scale_height(self.targ_T, self.species_dict[s], self.targ_g)
            depth_species = self.Delta_transit_depth(scaleheight_species, self.numH, self.targ_Rp, self.targ_Rs)
            scale = (depth_species/depth_model).decompose()
            print(scale)
            print(scaleheight_model, depth_model, scaleheight_species, depth_species)

            
            if self.tracer == False:

                refspec_model_depths = np.array(self.model_dict[self.species[0]]['depths'])
                wavelengths = self.model_dict[self.species[0]]['wavelengths'].value
                wave_center_indx = int(len(self.wave_centers)/2)
                indx = np.where(np.abs(wavelengths-self.wave_centers[wave_center_indx].value) == min(np.abs(wavelengths-self.wave_centers[wave_center_indx].value)))[0]
                mean_model_depths = np.mean(refspec_model_depths)#[indx] #np.mean(refspec_model_depths)
                model_depths = np.array(self.model_dict[s]['depths'])
                Delta_depths = model_depths - np.mean(model_depths)#mean_model_depths
                self.model_dict[s]['model_scaled'] = mean_model_depths + Delta_depths*scale


                refspec_model_depths_binned = np.array(self.model_dict[self.species[0]]['avg_depths'])
                mean_model_depths_binned = np.mean(refspec_model_depths_binned)
                self.model_dict[s]['avg_depths'] = np.zeros(len(self.wave_centers))
                for i in range(len(self.wave_centers)):
                    wavelengths = self.model_dict[self.species[0]]['wavelengths'].value
                    indx = np.where(np.abs(wavelengths-self.wave_centers[i].value) == min(np.abs(wavelengths-self.wave_centers[i].value)))[0]
                    self.model_dict[s]['avg_depths'][i] = np.array(self.model_dict[s]['depths'])[i]

                #refspec_model_depths_binned = np.array(self.model_dict[self.species[0]]['avg_depths'])
                #mean_model_depths_binned = np.mean(refspec_model_depths_binned)
                model_depths_binned = np.array(self.model_dict[s]['avg_depths'])
                Delta_depths_binned = model_depths_binned - mean_model_depths_binned
                self.model_dict[s]['model_scaled_binned'] = mean_model_depths_binned + Delta_depths_binned*scale

            else:

                refspec_model_depths = np.array(self.model_dict[self.species[0]]['depths'])
                wavelengths = self.model_dict[self.species[0]]['wavelengths'].value
                wave_center_indx = len(self.wave_centers)/2
                indx = np.where(np.abs(wavelengths-self.wave_centers[wave_center_indx].value) == min(np.abs(wavelengths-self.wave_centers[wave_center_indx].value)))[0]
                mean_model_depths = refspec_model_depths[indx] #np.mean(refspec_model_depths)
                Delta_depths = refspec_model_depths - mean_model_depths
                self.model_dict[s]['model_scaled'] = mean_model_depths + Delta_depths*scale
    '''
    def make_chisq_atmos(self, saveplot=''):
        
        self.chisq_atmos = {}

        for j in range(len(self.wave_centers)):
            for i in range(1000):
                gaussian_scatter = np.random.normal(0., self.phot_rms[j], 1)
                wavelengths = self.model_dict[self.species[0]]['wavelengths'].value
                indx = np.where(np.abs(wavelengths-self.wave_centers[j].value) == min(np.abs(wavelengths-self.wave_centers[j].value)))[0]
                simulated_data = self.model_dict[self.species[0]]['model_scaled'][indx].value + gaussian_scatter
                if i == 0: self.chisq_atmos[self.wave_centers[j].value] = []
                #self.chisq_atmos[self.wave_centers[j].value].append(np.sum((np.power((simulated_data - self.model_dict[self.species[0]]['model_scaled'][indx].value)/self.phot_rms[j], 2.))))
                self.chisq_atmos[self.wave_centers[j].value].append(simulated_data)
        
        bins=np.histogram(np.hstack((np.array(self.chisq_atmos[j.value]).flatten() for j in self.wave_centers)), bins=25)[1]
        for j in range(len(self.wave_centers)):
            plt.hist(np.array(self.chisq_atmos[self.wave_centers[j].value]).flatten(), bins, alpha=0.5, label='{0}'.format(str(self.wave_centers[j])))
            #plt.plot([self.wave_centers[j].value for i in range(1000)], self.chisq_atmos[self.wave_centers[j].value], 'o',  alpha=0.5)
        plt.legend(loc=0)
        plt.xlabel('Transit Depth [%]')
        plt.tight_layout()
        if saveplot != '':
            plt.savefig(self.targ_name+saveplot+'_chisq_atmos.png')
            plt.close()
        else:
            plt.show()
            plt.close()
    '''

    def make_chisq_atmos(self, saveplot=''):
        
        self.chisq_atmos = {}

        for i in range(1000):
            gaussian_scatter = np.random.normal(0., self.phot_rms, len(self.model_dict[self.species[0]]['avg_depths']))
            for s in self.species:
                simulated_data = self.model_dict[self.species[0]]['model_scaled_binned'] + gaussian_scatter
                if i == 0: self.chisq_atmos[s] = []
                self.chisq_atmos[s].append(np.sum(np.power(((simulated_data - self.model_dict[s]['model_scaled_binned'])/self.phot_rms), 2.)))

        bins=np.histogram(np.hstack((self.chisq_atmos[s] for s in self.species)), bins=25)[1]
        for s in self.species:
            plt.hist(self.chisq_atmos[s], bins, alpha=0.5, label='{0} atmosphere'.format(s, 'atmosphere'))
        plt.legend(loc=0)
        plt.xlabel(r'$\chi^2$')
        plt.tight_layout()
        if saveplot != '':
            plt.savefig(self.targ_name+saveplot+'_chisq_atmos.png')
            plt.close()
        else:
            plt.show()
            plt.close()


    def make_probability_plot(self, sigma, saveplot=''):
        # only tells the difference between two sets of points - automatically picks the first two
        M1 = np.array(self.chisq_atmos[self.wave_centers[0].value]).flatten()
        M2 = np.array(self.chisq_atmos[self.wave_centers[1].value]).flatten()
        DeltaM = []
        for i in range(1000):
            m1 = random.choice(M1)
            m2 = random.choice(M2)
            DeltaM.append(np.abs(m1-m2))

        var1 = np.var(M1)
        var2 = np.var(M2)
        sigma_quad = np.sqrt(var1 + var2)
        #line = sigma*sigma_quad
        
        plt.figure(figsize=(12,8))
        plt.hist(DeltaM/sigma_quad, bins=25, alpha=0.5, color='r')
        #plt.axvline(line, linestyle='--', lw=2., color='k', alpha=0.5)
        plt.xlabel('|Mred - Mblue|')
        plt.tight_layout()
        plt.show()            
        plt.close()


    def make_transmission_model(self, saveplot=''):
        plt.figure(figsize=(13, 7))
        xaxis = self.model_dict[self.species[0]]['wavelengths'].value
        xaxis_binned = self.model_dict[self.species[0]]['avg_wavelengths'].value
        linestyles = ['-', ':', '--', '-.']
        for i, s in enumerate(self.species):
            plt.plot(xaxis, self.model_dict[s]['model_scaled'].value*100., 'k', ls=linestyles[i%4], alpha=0.5, lw=3, label=r'${0}, \mu = {1},\ \chi^2 = $'.format(s, str(self.species_dict[s]))+str( '%.2f' %float(np.mean(self.chisq_atmos[s]))))
            print('sigma={0}'.format(get_significance(np.mean(self.chisq_atmos[s]), len(self.wave_centers)-1 )))

        for i in range(len(self.wave_centers)):
            gaussian_scatter = np.random.normal(0., self.phot_rms[i], 1)
            wavelengths = self.model_dict[self.species[0]]['wavelengths'].value
            indx = np.where(np.abs(wavelengths-self.wave_centers[i].value) == min(np.abs(wavelengths-self.wave_centers[i].value)))[0]
            simulated_data = self.model_dict[self.species[0]]['model_scaled'][indx].value + gaussian_scatter
            print(wavelengths[indx], simulated_data, self.phot_rms[i])
            plt.errorbar(wavelengths[indx], simulated_data*100., yerr=self.phot_rms[i]*100., fmt='o', color='k', alpha=0.7, lw=3, markersize='8', capsize=5)#, label=self.wave_centers[i])

        plt.xlim(self.wave_range.value)
        plt.xlabel('Wavelength [microns]', fontsize=16)
        plt.ylabel('Transi Depth [%]', fontsize=16)
        #plt.legend(loc=4, ncol=3, mode='expand', frameon=False)
        plt.legend(loc='best')
        plt.tight_layout()
        if saveplot != '':
            plt.savefig(self.targ_name+saveplot+'_transmission_model.png')
            plt.close()
        else:
            plt.show()
            plt.close()

def get_significance(chisq, dof):
    alpha = (1. - st.chi2.cdf(chisq, dof))/2.
    z = st.norm.ppf(1.-alpha)
    return z