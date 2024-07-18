import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from scipy.interpolate import interpn, interp1d
import astropy.units as u
from astropy.table import Table
import pickle
import os

homepath = '/home/hannah/code/astrotools/'

def save_new_spec(spec, outputdir=None, outputname=''):

	if outputdir == None: outputdir = os.getcwd() + '/'
	if outputname == '': outputname = 'interp_spec'

	ascii.write(spec, outputname+'.dat')


def interp_spec(model='SPHINX', Teff=3000, logg=4.0, logz=0.0, plot=False, save_spec=False, outputdir=None, outputname=''):

	print('Interpolating a new stellar spectrum from a grid of {} models'.format(model))

	if model == 'PHOENIX':

		Z = '-0.0'

		PHOENIXpath = homepath + f'StellarModels/PHOENIX-Z{Z}-HiRes/'
		wavelength = fits.getdata(homepath + 'StellarModels/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') * u.Angstrom

		# bulid up table of spectra
		def return_spectrum(in_temp, in_logg):
			specs = np.zeros((len(in_temp), len(in_logg), len(wavelength)))
			for i, temp in enumerate(in_temp):
				for j, logg in enumerate(in_logg):
					if temp < 10000: spec = fits.getdata(PHOENIXpath+'lte0{:4d}-{:1.2f}{}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(temp, logg, Z))
					else: spec = fits.getdata(PHOENIXpath+'lte{:4d}-{:1.2f}{}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(temp, logg, Z))

					specs[i][j] += spec

			return specs

		Temps = np.arange(2300, 5000, 100)
		loggs = np.arange(0, 6, 0.5)

		points = (Temps, loggs)
		specs = return_spectrum(Temps, loggs)


		# change the following Teff, logg:
		point = np.array([Teff, logg])
		interp_spec = interpn(points, specs, point)[0] * u.erg/u.cm**2/u.s/u.cm

	elif model == 'SPHINX':
		# Teff range: [3000, 4000]
		# logg range: [4.0, 5.5]
		# logZs range = [-1, 1]

		SPHINXpath = homepath + 'StellarModels/SPHINX_MODELS_MLT_1/SPECTRA/'
		wavelength = ascii.read(SPHINXpath+'Teff_2000.0_logg_4.0_logZ_+0.0_CtoO_0.3_spectra.txt')['col1'] * u.micron

		# bulid up table of spectra
		def return_spectrum(in_temp, in_logg, in_logZ):
			specs = np.zeros((len(in_temp), len(in_logg), len(in_logZ), len(wavelength)))
			for i, temp in enumerate(in_temp):
				for j, loggrav in enumerate(in_logg):
					for k, logZ in enumerate(in_logZ):
						if logZ < 0: sign = '-'
						else: sign = '+'

						filename = f'Teff_{temp}.0_logg_{loggrav}_logZ_{sign}{abs(logZ)}_CtoO_0.5_spectra'
						spec = ascii.read(SPHINXpath + filename + '.txt')['col2']

						
						specs[i][j][k] += spec

			return specs

		Temps = np.arange(2000, 4100, 100)
		loggs = np.arange(4.0, 5.75, 0.25)
		logZs = np.arange(-1, 1, 0.25)


		points = (Temps, loggs, logZs)

		try: specs = pickle.load(open(homepath + 'interp_SPHINX_spec.p', 'rb'))
		except:
			specs = return_spectrum(Temps, loggs, logZs)
			pickle.dump(specs, open(homepath + 'interp_SPHINX_spec.p', 'wb'))

		# change the following Teff, logg:
		point = np.array([Teff, logg, logz])

		interp_spec = interpn(points, specs, point)[0] * u.Watt / u.m**2 / u.m

	elif model == 'BT-SETTL':

		metal = '0.0'
		alpha = '-0.0'
		#lte022.0-4.0-0.0a+0.0.BT-Settl.spec.7.dat
		BTSETTLpath = homepath+'StellarModels/bt-settl-cifist/'
		wavelength = ascii.read(BTSETTLpath + 'lte020.0-3.5-0.0a+0.0.BT-Settl.spec.7.dat.txt', names=['wave', 'flux'])['wave'] * u.Angstrom
		goodinds = (wavelength.value < (50*u.micron).to(u.Angstrom).value)
		wavelength = wavelength[goodinds]

		# bulid up table of spectra
		def return_spectrum(in_temp, in_logg):
			specs = np.zeros((len(in_temp), len(in_logg), len(wavelength)))
			for i, temp in enumerate(in_temp):
				print('temp = ', temp)
				for j, logg in enumerate(in_logg):
					print('logg = ', logg)
					model = ascii.read(BTSETTLpath+'lte0{:d}'.format(temp)[:-2]+'.'+'{:d}'.format(temp)[2]+'-{:1.1f}{}a+{}.BT-Settl.spec.7.dat.txt'.format(logg, alpha, metal), names=['wave', 'flux'])
					wave, spec = model['wave'], model['flux']
					#else: spec = fits.getdata(PHOENIXpath+'lte{:4d}-{:1.2f}{}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(temp, logg, Z))

					# cut off spectrum at 50 microns becuase otherwise grid is too bit
					goodinds = (wave < (50*u.micron).to(u.Angstrom).value)
					wave, spec = wave[goodinds], spec[goodinds]

					if len(spec) != len(wavelength):
						f = interp1d(wave, spec, fill_value='extrapolate')
						spec = f(wavelength.value)

					specs[i][j] += spec

			return specs

		#Temps = np.hstack([np.arange(2000, 2400, 50), np.arange(2400, 6100, 100)])
		#Temps = np.arange(2400, 6100, 100)
		Temps = np.arange(2400, 3500, 100)
		loggs = np.arange(3.5, 6, 0.5)

		points = (Temps, loggs)
		specs = return_spectrum(Temps, loggs)


		# change the following Teff, logg:
		point = np.array([Teff, logg])
		interp_spec = interpn(points, specs, point)[0] * u.erg/u.cm**2/u.s/u.Angstrom


	interp_spec = interp_spec.to(u.erg/u.cm**2/u.s/u.Angstrom)
	wavelength = wavelength.to(u.Angstrom)


	if plot:
		plt.figure(figsize=(12, 5))
		# get closest spectrum:
		temp_ind = np.argmin(abs(Temps - Teff))
		logg_ind = np.argmin(abs(loggs - logg))
		logz_ind = np.argmin(abs(logZs - logz))
		if logZs[logz_ind]<0: sign='-'
		else: sign='+'

		closest_spec_name = f'Teff_{Temps[temp_ind]}.0_logg_{loggs[logg_ind]}_logZ_{sign}{abs(logZs[logz_ind])}_CtoO_0.5_spectra'
		closest_spec_data = ascii.read(SPHINXpath + closest_spec_name + '.txt')
		closest_spec_wave = (closest_spec_data['col1'] * u.micron).to(u.Angstrom)
		closest_spec_flux = (closest_spec_data['col2'] * u.Watt / u.m**2 / u.m).to(u.erg/u.cm**2/u.s/u.Angstrom)
		plt.plot(closest_spec_wave.to(u.micron), closest_spec_flux, alpha=0.7, label=f'SPINX spec:\nTeff={Temps[temp_ind]}\nlogg={loggs[logg_ind]}\nlogz={logZs[logz_ind]}')
		plt.plot(wavelength.to(u.micron), interp_spec, alpha=0.7, label='Interpolated')

		plt.grid(alpha=0.4)
		plt.xlabel('Wavelength ($\mu$m)')
		plt.ylabel('Flux Density (erg/cm$^2$/s/$\AA$)')
		plt.tight_layout()
		plt.legend(loc='best')

		plt.show()

	#ascii.write([wavelength, interp_spec], 'interpolated_PHOENIX_Z{}-Teff-{}_logg-{}.dat'.format(Z, point[0], point[1]), names=['Wavelength(A)', 'Flux(erg/cm2/s/A)'], overwrite=True)
	new_spec = Table()
	new_spec['Wavelength(A)'] = wavelength
	new_spec['Flux(erg/cm2/s/A)'] = interp_spec

	if save_spec: save_new_spec(new_spec, outputdir, outputname)

	return new_spec



####### figure out how CtoO changes for these stellar spectra

#Temp = '4000'
#logg = '4.0'
#logZ = '+0.0'
#CtoO = ['0.3', '0.5', '0.7', '0.9']


#plt.figure(figsize=(15, 5))
#for frac in CtoO:
#	model = ascii.read(f'./StellarModels/SPHINX_MODELS_MLT_1/SPECTRA/Teff_{Temp}.0_logg_{logg}_logZ_{logZ}_CtoO_{frac}_spectra.txt')
#	plt.plot(model['col1'], model['col2'])

#plt.show()
