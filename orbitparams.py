# useful tools to calculate various orbital parameters

import numpy as np
import astropy.units as u
import astropy.constants as c

#takes a/Rs and imbact parameter b; returns inclination in degress
def inc(a_Rs, b):
    '''
    takes: scaled orbital distance a_Rs []
           impact paramter b []
    returns: inclination i [degrees]
    '''
    return np.degrees(np.acos(b*(1./a_Rs)))

def impact(a_Rs, i):
    '''
    takes: scaled orbital distance a_Rs []
           inclination i [degrees]
    returns: imact parameter b []
    '''
    return a_Rs * np.cos(np.radians(i))

def Tdur_occ_ecc(P, Rp_Rs, a_Rs, i=90, e=0, omega=90):
  

    #assert b != None or i != None

    b_occ = a_Rs * np.cos(np.radians(i)) * ((1 - e**2) / (1 - e*np.sin(np.radians(omega))))

    mult_factor = np.sqrt(1 - e**2) / (1 - e*np.sin(np.radians(omega)))

    value = np.sqrt((1 + Rp_Rs)**2 - b_occ**2) / a_Rs / np.sin(np.radians(i))
    
    return P/np.pi * np.arcsin(value) * mult_factor

def Tdur(P, Rp_Rs, a_Rs, b=None, i=None):
    '''
    T14
    takes: period P [days]
           radius ratio Rp_Rs []
           impact parameter b []
           scaled orbital distance a_Rs []
           inclination i [deg]
    returns: duration [days]

    both b and i cannot be None
    '''

    assert b != None or i != None


    if i==None: i = np.radians(inc(a_Rs, b))
    elif b == None: 
        b = impact(a_Rs, i)
        i = np.radians(i)

    value = np.sqrt((1 + Rp_Rs)**2 - b**2) / a_Rs / np.sin(i)
    return P/np.pi * np.arcsin(value)

def Tdur_NEA(targ):
    '''
    Calculate transit (or eclipse) duration from dictionary formatted like NASA Exoplanet Archive
    '''
    if targ['pl_ratror'] == None:
        print("Calculating Rp/Rs from separate Rp and Rs")
        Rp_Rs = ((targ['pl_rade']*u.R_earth)/(targ['st_rad']*u.R_sun)).decompose().value
    else: Rp_Rs = targ['pl_ratror']
    if targ['pl_ratdor'] == 0.0:
        print("Calculating a/Rs from separate a and Rs")
        a_Rs = ((targ['pl_orbsmax']*u.AU)/(targ['st_rad']*u.R_sun)).decompose().value
    else: a_Rs = targ['pl_ratdor']
    tdur = Tdur(P=targ['pl_orbper']*u.day, 
                Rp_Rs=Rp_Rs,
                a_Rs=a_Rs,
                i = targ['pl_orbincl']
                ) # event duration
    
    return tdur

def T23(P, Rp_Rs, a_Rs, b,  i=None):
    '''
    T23
    takes: period P [days]
           radius ratio Rp_Rs []
           impact parameter b []
           scaled orbital distance a_Rs []
           inclination i [rad] (optional)
    returns: duration [days]
    '''
    
    if i==None: i = np.radians(inc(a_Rs, b))
    value = np.sqrt((1 - Rp_Rs)**2 - b**2) / a_Rs / np.sin(i)
    return P/np.pi * np.arcsin(value)

def Tdur_ecc(P, Rp_Rs, a_Rs, e, w, b=None, i=None):

    '''
    T14
    takes: period P [days]
           radius ratio Rp_Rs []
           impact parameter b []
           scaled orbital distance a_Rs []
           inclination i [deg]
           eccentricity e
           argument of periastron w [radians]
    returns: duration [days]

    both b and i cannot be None
    '''

    assert b != None or i != None

    if i==None: i = np.radians(inc(a_Rs, b))
    elif b == None: 
        b = impact(a_Rs, i)
        i = np.radians(i)

    ecc_approx = np.sqrt(1 - e**2) / (1 + e*np.sin(w))

    value = np.sqrt((1 + Rp_Rs)**2 - b**2) / a_Rs / np.sin(i)

    return P/np.pi * np.arcsin(value) * ecc_approx




def Teq(Teff, A, a_Rs, f=1/4):
  '''
  takes: effective temperature of the star [K]
         Albedo of the planet (from 0 - 1)
         a_Rs is scaled orbital distance a/Rs
         f = 1/4 for equlibrium temperature (assuming uniform redistribution). change to 2/3 to approximate instant-reradiation (Koll et al., 2019)
  returns: equilibrium temperature in Kelvin
  '''

  return Teff * (1 - A)**(1/4) * np.sqrt(1/a_Rs) * f**(1/4)

def semimajoraxis(P, Ms, P_unc, Ms_unc):
    '''
    takes: period P
           stellar mass M

    returns: semi-major axis, uncertaintiy in semi-major axis
    '''

    a = (c.G * Ms * P**2 / (4 * np.pi**2))**(1/3)
    da_dM = ((c.G * P**2)/(4 * np.pi**2))**(1/3) * (1/3) * Ms**(-2/3)
    da_dP = ((c.G * Ms)/(4 * np.pi**2))**(1/3) * (2/3) * P**(-1/3)
    da = np.sqrt((da_dM**2)*(Ms_unc**2) + (da_dP**2)*(P_unc**2))
    return a.to(u.AU), da.to(u.AU)

def instellation(Teff, Teff_unc, Rs, Rs_unc, a, a_unc):
    '''
    # put the units in using astropy.units
    takes: stellar effective temperature Teff
           stellar effective temperature Teff uncertainty
           stellar radius Rs 
           stellra radius Rs uncertainty
           semi-major axis a
           semi-major axis uncertainty a unc
    returns: insolation in earth insolations
    '''
    Splanet = c.sigma_sb * Teff**4 * Rs**2 * np.pi / a**2
    Searth = c.sigma_sb * (5777 * u.K)**4 * (1 * u.Rsun)**2 * np.pi / (1 * u.AU)**2
    dS_dTeff = c.sigma_sb * np.pi * (Rs/a)**2 * 4 * Teff**3
    dS_dRs = c.sigma_sb * Teff**4 * np.pi * 2 * Rs / a**2
    dS_da = c.sigma_sb * Teff**4 * Rs**2 * np.pi * (-2) * a**(-3)
    dS = np.sqrt((dS_dTeff**2 * Teff_unc**2) + (dS_dRs**2 * Rs_unc**2) + (dS_da**2 * a_unc**2))
    return (Splanet/Searth).decompose(), (dS/Searth).decompose()



def F_bol(L_bol, L_bol_err, dist, dist_err):

    F = L_bol/(4*np.pi*dist**2)

    dF_dL = 1 / (4*np.pi*dist**2)
    dF_ddist = -L_bol / (2 * np.pi * dist**3)

    sigF = np.sqrt( (dF_dL**2 * L_bol_err**2) + (dF_ddist**2 * dist_err**2) )

    return F.to(u.erg/u.cm**2/u.s), sigF.to(u.erg/u.cm**2/u.s)


def rho(Mp, Rp):
    '''
    calculate the planetary density on g/cm^3
    '''
    Vp = 4/3 * np.pi * Rp**3
    return Mp/Vp