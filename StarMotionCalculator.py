from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

class StarMotionCalculator:
    def __init__(self, star_name, coords_string, epoch, pmra, pmdec, parallax, radial_velocity, t):
        """
        Initialize the calculator with star properties and elapsed time.

        Parameters:
        - star_name: str, name of the star whose motion you are calculating
        - coords_string: str, coordinates of the star (e.g., '12h34m56.7s -45d12m34s')
        - epoch: float, timestamp corresponding to the coordinates [yr]
        - pmra: float, proper motion in right ascension [mas/yr]
        - pmdec: float, proper motion in declination [mas/yr]
        - parallax: float, parallax [mas]
        - radial_velocity: float, radial velocity of target [km/s]
        - t: float, elapsed time since reference epoch [yr]
        """
        self.star_name = star_name
        self.t = t
        self.epoch = epoch
        self.c = SkyCoord(coords_string, unit=(u.hourangle, u.deg), frame='icrs')

        # Extract Gaia parameters
        self.mu_ra = pmra / 1e3  # Proper motion in RA [as/yr]
        self.mu_dec = pmdec / 1e3  # Proper motion in Dec [as/yr]
        self.d = 1 / (parallax / 1e3)  # Distance [pc]
        self.v_r = radial_velocity  # Radial velocity [km/s]

    def calculate_new_position(self):
        """
        Calculate the new position of the star after elapsed time.

        Returns:
        - c_new: SkyCoord, new coordinates of the star
        """
        # Initial RA and Dec in radians
        ra = self.c.ra.deg * np.pi / 180
        dec = self.c.dec.deg * np.pi / 180

        # Convert to transverse (linear) velocities
        v_ra = self.mu_ra * self.d * 4.74       # [km/s]
        v_dec = self.mu_dec * self.d * 4.74     # [km/s]

        # Convert to cartesian velocities
        v_x = ((self.v_r * np.cos(dec) * np.cos(ra)) - (v_ra * np.sin(ra)) - (v_dec * np.sin(dec) * np.cos(ra))) / 977780  # [pc/yr]
        v_y = ((self.v_r * np.cos(dec) * np.sin(ra)) + (v_ra * np.cos(ra)) - (v_dec * np.sin(dec) * np.sin(ra))) / 977780  # [pc/yr]
        v_z = ((self.v_r * np.sin(dec)) + (v_dec * np.cos(dec))) / 977780                                              # [pc/yr]

        # Initial cartesian position
        x0 = self.d * np.cos(dec) * np.cos(ra)
        y0 = self.d * np.cos(dec) * np.sin(ra)
        z0 = self.d * np.sin(dec)

        # Updated position
        xt = x0 + v_x * self.t  # [pc]
        yt = y0 + v_y * self.t  # [pc]
        zt = z0 + v_z * self.t  # [pc]

        # Convert back to spherical coordinates
        dxy = np.sqrt(xt**2 + yt**2)  # [pc]
        dec_new = np.arctan(zt / dxy) * 180 / np.pi  # [deg]
        ra_new = np.arctan(yt / xt) * 180 / np.pi    # [deg]

        if yt < 0 and xt < 0:
            ra_new += 180.  # RA is in the 3rd quadrant
        elif yt > 0 and xt < 0:
            ra_new += 180.  # RA is in the 3rd quadrant

        # Return new SkyCoord object
        c_new = SkyCoord(ra_new, dec_new, unit='deg', frame='icrs')
        return c_new

    def get_new_position(self):
        """
        Print the old and new position of the star in a readable format.
        """
        print(f"Starting position of {self.star_name} in {(self.epoch)}: {self.c.to_string('hmsdms')}")

        c_new = self.calculate_new_position()
        print(f"New position of {self.star_name} in {(self.epoch + self.t)}: {c_new.to_string('hmsdms')[0]}")
        return c_new

# Example usage:
# gaia_coords = {
#     'pmra': ...,  # Proper motion in RA [mas/yr]
#     'pmdec': ...,  # Proper motion in Dec [mas/yr]
#     'parallax': ...,  # Parallax [mas]
#     'radial_velocity': ...,  # Radial velocity [km/s]
#     'input_star_name': ['StarName'],
#     'ref_epoch': 2016.0,  # Reference epoch [yr]
# }
# coords_string = '12h34m56.7s -45d12m34s'
# t = 6

# star_calculator = StarMotionCalculator(coords_string, gaia_coords, t)
# star_calculator.print_new_position()
