#write new userInput.in files and run Exo_Transmit

import numpy as np
import astropy.units as u
import astropy.constants as c
import os
import subprocess

# THIS IF FOR LHS1140b
Rp_base = 7398637.76 * u.m
Rs = 143970156.0 * u.m
g = 14.05
TPfile = 't_p_800K'
EOSfile = 'eos_1Xsolar_cond'
Outputfile = 'LHS3844'

'''
# this is to adjust the 1 bar planet radius when matching transmission models to data
Rp_adjust = [-0.12, -0.13, -0.14, -0.15, -0.16]   # add or subtract by this percentage of Rp

for a in Rp_adjust:
    print('    TPprofile:', TPfile)
    print('    EOSfile:', EOSfile)
    print('    adjust by', a*100, '%')

    inputfile = open('/home/hannah/programs/Exo_Transmit/userInput.in', 'w')
    inputfile.write('userInput.in -\n')
    inputfile.write('Formatting here is very important, please put the instructed content on the instructed line!\n')
    inputfile.write('Exo_Transmit home directory:\n')
    inputfile.write('/home/hannah/programs/Exo_Transmit\n')
    inputfile.write('Temperature-Pressure data file:\n')
    inputfile.write('/T_P/'+TPfile+'.dat\n')
    inputfile.write('Equation of State file:\n')
    inputfile.write('/EOS/'+EOSfile+'.dat\n')
    inputfile.write('Output file:\n')
    inputfile.write('/Spectra/'+Outputfile+'_'+str(a)+'.dat\n')
    inputfile.write('Planet surface gravity (in m/s^-2):\n')
    inputfile.write('12.4\n')
    inputfile.write('Planet radius (in m): -- This is the planet\'s radius at the base of the atmosphere (or at the cloud top for cloudy calculations)\n')
    Rp_adjusted = (Rp_base + a*Rp_base).value
    inputfile.write(str(Rp_adjusted)+'\n')
    inputfile.write('Star radius (in m):\n')
    inputfile.write(str(Rs.value)+'\n')
    inputfile.write('Pressure of cloud top (in Pa), -- or leave at 0.0 if you want no cloud calculations.\n')
    inputfile.write('0.0\n')
    inputfile.write('Rayleigh scattering augmentation factor -- default is 1.0.  Can be increased to simulate additional sources of scattering.  0.0 turns off Rayleigh scattering.\n')
    inputfile.write('1.0\n')
    inputfile.write('End of userInput.in (Do not change this line!)\n')

    inputfile.close()

    os.chdir('/home/hannah/programs/Exo_Transmit')
    os.system('./Exo_Transmit')
    os.chdir('/home/hannah/code/astrotools')
    #subprocess.call(['/home/hannah/programs/Exo_Transmit/Exo_Transmit'])
    #./Exo_Transmit


'''
# this is to test which moelcules contribute most to the 1Xsolar spectrum
molecule = ['CH4', 'CO2', 'CO', 'H2O', 'NH3', 'O2', 'O3', 'C2H2', 'C2H4', 'C2H6', 'H2CO', 'H2S', 'HCl', 'HCN', 'HF', 'MgH', 'N2', 'NO', 'NO2', 'OCS', 'OH', 'PH3', 'SH', 'SiH', 'SiO', 'SO2', 'TiO', 'VO', 'Na', 'K']
# pick the scaling value of a you desire!
a = 1
for i, m in enumerate(molecule):

    print 'writing user input file with'
    print '    TPprofile:', TPfile
    print '    EOSfile:', EOSfile
    print '    molecule:', m

    zeros = np.zeros(len(molecule))
    zeros[i] = 1

    chemfile = open('/home/hannah/programs/Exo_Transmit/selectChem.in', 'w')
    chemfile.write('For each gas, place a 1 after the equals sign if the gas opacity will be present in your transmission calculations, and 0 if not. Do not change the order of the gases in this file!\n')
    for j, n in enumerate(molecule):
        #print str(n + ' = ' + str(zeros[j] + '\n')
        chemfile.write(n + ' = ' + str(int(zeros[j])) + '\n')
    chemfile.write('Scattering = 1\n')
    chemfile.write('Collision Induced Absorption = 1\n')
    chemfile.write('Do not delete this line!')
    chemfile.close()


    inputfile = open('/home/hannah/programs/Exo_Transmit/userInput.in', 'w')
    inputfile.write('userInput.in -\n')
    inputfile.write('Formatting here is very important, please put the instructed content on the instructed line!\n')
    inputfile.write('Exo_Transmit home directory:\n')
    inputfile.write('/home/hannah/programs/Exo_Transmit\n')
    inputfile.write('Temperature-Pressure data file:\n')
    inputfile.write('/T_P/tp_'+TPfile+'.dat\n')
    inputfile.write('Equation of State file:\n')
    inputfile.write('/EOS/'+EOSfile+'.dat\n')
    inputfile.write('Output file:\n')
    inputfile.write('/Spectra/'+TPfile+'_'+m+'.dat\n')
    inputfile.write('Planet surface gravity (in m/s^-2):\n')
    inputfile.write(g+'\n')
    inputfile.write('Planet radius (in m): -- This is the planet\'s radius at the base of the atmosphere (or at the cloud top for cloudy calculations)\n')
    Rp_adjusted = (Rp_base + np.sign(a)*Rp_base*abs(a)).value
    inputfile.write(str(Rp_adjusted)+'\n')
    inputfile.write('Star radius (in m):\n')
    inputfile.write(Rs.value+'\n')
    inputfile.write('Pressure of cloud top (in Pa), -- or leave at 0.0 if you want no cloud calculations.\n')
    inputfile.write('0.0\n')
    inputfile.write('Rayleigh scattering augmentation factor -- default is 1.0.  Can be increased to simulate additional sources of scattering.  0.0 turns off Rayleigh scattering.\n')
    inputfile.write('1.0\n')
    inputfile.write('End of userInput.in (Do not change this line!)\n')

    inputfile.close()

    os.chdir('/home/hannah/programs/Exo_Transmit')
    os.system('./Exo_Transmit')
    os.chdir('/home/hannah/code/astrotools')
