import numpy as np
from decimal import Decimal

# The EOS file that this code spits out should look identical in format to the eos files provided in the Exo_Transmit code

# change the name (or path) to whatever you want! I tihnk the .dat ending is necessary though
eos_name = 'eos_01H2O_99H2.dat'
f = open(eos_name, 'w')

# set the fractional proportion of each gas you want in your atmosphere
# make sure all your fractions add up to 1.0!!
C = 0.
CH4 = 0.
CO = 0.
COS = 0.
CO2 = 0.
C2H2 = 0.
C2H4 = 0.
C2H6 = 0.
H = 0.
HCN = 0.
HCl = 0.
HF = 0.
H2 = 0.99
H2CO = 0.
H2O = 0.01
H2S = 0.
He = 0.
K = 0.
MgH = 0.
N = 0.
N2 = 0.
NO2 = 0.
NH3 = 0.
NO = 0.
Na = 0.
O = 0.
O2 = 0.
O3 = .0
OH = 0.
PH3 = 0.
SH = 0.
SO2 = 0.
SiH = 0.
SiO = 0.
TiO = 0.
VO = 0.

# no need to change anything here
header = 'T		P		C		CH4		CO		COS		CO2		C2H2		C2H4		C2H6		H		HCN		HCl		HF		H2		H2CO		H2O		H2S		He		K		MgH		N		N2		NO2		NH3		NO		Na		O		O2		O3		OH		PH3		SH		SO2		SiH		SiO		TiO		VO		'

# or here
molecules = np.array([C, CH4, CO, COS, CO2, C2H2, C2H4, C2H6, H, HCN, HCl, HF, H2, H2CO, H2O, H2S, He, K, MgH, N, N2, NO2, NH3, NO, Na, O, O2, O3, OH, PH3,	SH, SO2, SiH, SiO, TiO, VO])

# or here
# there are the pressure and temperature ranges that Eliza calculates, which should be plenty for any case (not sure how it works if you change these...)
T = np.flipud(np.arange(1e2, 3.1e3, 100))
P = np.flipud(np.logspace(-4, 8, 13))

# write the EOS file
f.write(header)
f.write('\n' + '\n')

for p in P:
    f.write('%e' % Decimal(p) + '\n' + '\n')
    for t in T:
        f.write('%e' % Decimal(t) + '\t' + '%e' % Decimal(p) + '\t')
        for m in molecules: f.write('%e' % Decimal(m) + '\t')
        f.write('\n')
    f.write('\n')

f.close()


