import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def apply_proper_motion(c, mu_ra, mu_dec, d, t, v_r=0):
    """
    Propagate a sky coordinate forward in time using proper motion.

    Parameters
    ----------
    c : SkyCoord
        Reference position (epoch of catalog measurement).
    mu_ra : float
        Proper motion in RA [mas/yr] (includes cos(dec) factor).
    mu_dec : float
        Proper motion in Dec [mas/yr].
    d : float
        Distance [pc].
    t : float
        Elapsed time since reference epoch [yr].
    v_r : float, optional
        Radial velocity [km/s]. Default 0.

    Returns
    -------
    c_new : SkyCoord
        Propagated position in ICRS.
    """
    ra = c.ra.deg * np.pi / 180
    dec = c.dec.deg * np.pi / 180

    # convert mu to transverse (linear) velocities [km/s]
    v_ra = (mu_ra / 1000) * d * 4.74
    v_dec = (mu_dec / 1000) * d * 4.74

    # convert to cartesian velocities [pc/yr]
    v_x = ((v_r*np.cos(dec)*np.cos(ra)) - (v_ra*np.sin(ra)) - (v_dec*np.sin(dec)*np.cos(ra))) / 977780
    v_y = ((v_r*np.cos(dec)*np.sin(ra)) + (v_ra*np.cos(ra)) - (v_dec*np.sin(dec)*np.sin(ra))) / 977780
    v_z = ((v_r*np.sin(dec)) + (v_dec*np.cos(dec))) / 977780

    # propagate cartesian position
    x0 = d * np.cos(dec) * np.cos(ra)
    y0 = d * np.cos(dec) * np.sin(ra)
    z0 = d * np.sin(dec)

    xt = x0 + v_x * t
    yt = y0 + v_y * t
    zt = z0 + v_z * t

    # convert back to ra/dec [deg]
    dxy = np.sqrt(xt**2 + yt**2)
    dec_new = np.arctan(zt / dxy) * 180 / np.pi
    ra_new = np.arctan2(yt, xt) * 180 / np.pi % 360

    return SkyCoord(ra_new, dec_new, unit='deg', frame='icrs')


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
#c = SkyCoord('10 14 51.8945 -47 09 24.647', unit=(u.hourangle, u.deg), frame='fk5') # GJ 1132
#c = SkyCoord('14.2612 +19.1873', unit=(u.hour, u.deg), frame='fk5')
#c1 = SkyCoord('00 42 27.94 -15 32 11.4', unit=(u.hourangle, u.deg), frame='fk4')
#c = SkyCoord('00 44 59.33 -15 16 16.66', unit=(u.hourangle, u.deg), frame='icrs') # LHS 1140
#c = SkyCoord('00 45 03.0934 -15 18 43.869', unit=(u.hourangle, u.deg), frame='icrs') # TYC 5847-1671-1 (LHS 1140 comparison)
#c = SkyCoord('22:41:58.1173365142 -69:10:08.320922853', unit=(u.hourangle, u.deg), frame='icrs') # LHS 3844
#c = SkyCoord('03:01:51.39 -16:35:36.1', unit=(u.hourangle, u.deg), frame='icrs') # LTT 1445A  #ep=2000 (Winters)
#c = SkyCoord('03:01:51.04 -16:35:31.0', unit=(u.hourangle, u.deg), frame='icrs') # LTT 1445 BC ep=2000 (Winters; Henry+ (2018))
#c = SkyCoord('03:01:50.98 -16:35:40.32', unit=(u.hourangle, u.deg), frame='icrs') # LTT 1445A ep=2016 (Gaia Dr3)
#mu_ra = -0.36997184335074684 # pmra [as/yr]
#mu_dec = -0.267931311928784 # pmdec [as/yr]
#d = 6.863785972580628 # distance [pc]
#t = 10 # elapsed time since reference epoch [yr]
#c = SkyCoord('18:57:39.34 +53:30:33.3', unit=(u.hourangle, u.deg), frame='icrs') # WD 1856+354
#c = SkyCoord('18:57:37.911 +53:31:12.92', unit=(u.hourangle, u.deg), frame='icrs') # M dwarf near WD 1856+354
#c = SkyCoord('12:47:55.53 09:44:57.68', unit=(u.hourangle, u.deg), frame='icrs') # GJ 486 ep=2016 (Gaia EDR3)
#c = SkyCoord('12:47:55.57 09:44:57.91', unit=(u.hourangle, u.deg), frame='icrs') # GJ 486 ep=2015.5 (Gaia DR2)
#c = SkyCoord('12:47:59.18 09:40:07.74', unit=(u.hourangle, u.deg), frame='icrs') # GJ 486 comparison star ep=2016 (Gaia EDR3)
#mu_ra = -1.008267   # proper motion ra [as/yr]
#mu_dec = -0.460034  # proper motion dec [as/yr]
#d = 8.079134       # distance [pc]
#t = 6.16         # elapsed time since position was measured [yrs]
#c_new = apply_proper_motion(c, mu_ra, mu_dec, d, t)

