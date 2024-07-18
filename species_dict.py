import numpy as np

'''
Here are some species and their molecular weights; 
underscore format is helpful if you're ever trying to do some fancy LaTeX labels
'''

def create_species_dict():

    species_dict = {}

    species_dict['H_2'] = 2
    species_dict['N_2'] = 28
    species_dict['CH_4'] = 16
    species_dict['H_2O'] = 18
    species_dict['H2O'] = 18
    species_dict['h2o'] = 18
    species_dict['CO_2'] = 44
    species_dict['CO2'] = 44
    species_dict['CO2_cloudy'] = 44
    species_dict['co2'] = 44
    species_dict['O_2'] = 32
    species_dict['O2'] = 32
    species_dict['O_3'] = 48
    species_dict['O3'] = 48
    species_dict['O_2_nocollision'] = 32
    species_dict['CO'] = 28
    species_dict['flat'] = 10000
    species_dict['solar'] = 2.36
    species_dict['Solar'] = 2.36

    species_dict['98N_2_02O_3'] = 0.98 * species_dict['N_2'] + 0.02 * species_dict['O_3']
    species_dict['999N_2_001O_3'] = 0.999 * species_dict['N_2'] + 0.001 * species_dict['O_3']
    species_dict['7N_2_299O_2_001O_3'] = 0.7 * species_dict['N_2'] + 0.299 * species_dict['O_2'] + 0.001 * species_dict['O_3']
    species_dict['999O_2_001O_3'] = 0.999 * species_dict['N_2'] + 0.001 * species_dict['O_3']
    species_dict['999999N_2_000001O_3'] = 0.999999 * species_dict['N_2'] + .000001 * species_dict['O_3']
    species_dict['7N_2_299H_2O_001O_3'] = 0.7 * species_dict['N_2'] + .299 * species_dict['H_2O'] + .001 * species_dict['O_3']
    species_dict['29H_2O_7O_2_01O_3'] = .29 * species_dict['H_2O']  + 0.7 * species_dict['O_2'] + .01 * species_dict['O_3']
    species_dict['30H_2O_7O_2'] = .3 * species_dict['H_2O']  + 0.7 * species_dict['O_2']

    # models from Eliza Kempton, specifically for GJ 1132b
    species_dict['1Xsolar'] = 2.36
    species_dict['10Xsolar'] = 2.51
    species_dict['100Xsolar'] = 4
    species_dict['1000Xsolar'] = 13
    species_dict['1p0H2O'] = 18
    species_dict['0p5H2O'] = 10
    species_dict['0p1H2O'] = 3.6
    species_dict['0p01H2O'] = 2.16

    species_dict['1Xsolar'] = 2.36
    species_dict['1xsolar_noCIA'] = 2.36
    species_dict['1xsolar'] = 2.36
    species_dict['1xsolar_test'] = 2.36
    species_dict['1xsolar_change'] = 2.36
    species_dict['1Xsolar_clear'] = 2.36
    species_dict['1Xsolar_10mbar'] = 2.36
    species_dict['1Xsolar_1mbar'] = 2.36
    species_dict['1Xsolar_0p1mbar'] = 2.36
    species_dict['1Xsolar_0p01mbar'] = 2.36

    species_dict['10xsolar'] = 2.60
    species_dict['100xsolar'] = 5.2

    return species_dict

