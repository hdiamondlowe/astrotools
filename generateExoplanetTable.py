import pyvo as vo
from astropy.io import ascii

def generateTransitingExoTable(outputpath=".", sy_dist_upper=25, st_rad_upper=0.59, st_mass_upper=0.58, pl_rade_upper=1.85, pl_per_upper=30):
    """
    This code will generate a table of transiting planets from the NASA Exoplanet Archive.
    It is written for pulling lists of small planets, but parameters can be adjusted as needed.
    The default is to pull planets from the pscomppars table, in which each planet has one entry.

    Dependencies:

    astropy -- because duh everything needs astropy (but honestly only used for writing out table here)
    pyvo    -- https://pyvo.readthedocs.io/en/latest/
               PyVO lets you find and retrieve astronomical data available from archives that support standard IVOA virtual observatory service protocols:
                -Table Access Protocol (TAP) – accessing source catalogs using sql-ish queries.
                -Simple Image Access (SIA) – finding images in an archive.
                -Simple Spectral Access (SSA) – finding spectra in an archive.
                -Simple Cone Search (SCS) – for positional searching a source catalog or an observation log.
                -Simple Line Access (SLAP) – finding data about spectral lines, including their rest frequencies.

    Inputs:
    outputpath     -- Location of where to put the output data. Be careful as this will be automatically overwritten.
    sy_dist_upper  -- upper limit on the system distance (pc)
    st_rad_upper   -- upper limit on the stellar radius (solar radii)
    st_mass_upper  -- upper limit on the stellar mass (solar mass)
    pl_rade_upper  -- upper limit on the planet radius (in earth radii)
    pl_per_upper   -- upper limit on the planet's orbital period (days)

    * default upper limits on stellar mass and radius are M dwarf upper limits
      default upper limit on planet radius is terrestrial planet upper limit

    Returns:
    table of transiting exoplant systems from the NASA Exoplanet Archive

    Notes:
    These parameters and constraints are most applicable to me, but this code can be borrowed or ammened to add other keys or make different cuts in the data. 

    """

    print('Accessing NASA Exoplanet Archive')
    access_url = "https://exoplanetarchive.ipac.caltech.edu/TAP"
    service = vo.dal.TAPService(access_url)
    result = service.search(f"SELECT hostname,pl_name,sy_dist,rastr,decstr,sy_vmag,sy_jmag,sy_kmag,st_mass,st_rad,st_raderr1,st_raderr2,st_teff,st_tefferr1,st_tefferr2,st_logg,st_lum,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_rade,pl_radeerr1,pl_radeerr2,pl_bmasse,pl_bmasseerr1,pl_bmasseerr2,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbsmax,pl_orbsmaxerr1,pl_orbsmaxerr2,pl_orbincl,pl_orbeccen,pl_orbeccenerr1,pl_orbeccenerr2,pl_ratror,pl_ratrorerr1,pl_ratrorerr2,pl_ratdor,pl_ratdorerr1,pl_ratdorerr2,tran_flag,rv_flag FROM pscomppars WHERE pl_rade<={pl_rade_upper} and sy_dist<={sy_dist_upper} and st_rad<={st_rad_upper} and st_mass<={st_mass_upper} and pl_orbper<={pl_per_upper} and tran_flag=1 ORDER BY pl_name")

    print('Table of transiting exoplanets retrieved')
    transitingPlanetSystems = result.to_table()

    print('Writing table to {}'.format(outputpath+'/NASAExoArchive_TransitingExoplanetTable.dat'))
    ascii.write(transitingPlanetSystems, outputpath+'/NASAExoArchive_TransitingExoplanetTable.dat', overwrite=True) 

    return transitingPlanetSystems

def generateExoTable(outputpath=".", sy_dist_upper=25, st_rad_upper=0.59, st_mass_upper=0.58, pl_per_upper=30):
    """
    This code will generate a table of planets, both transiting and non-transiting from the NASA Exoplanet Archive.
    It is written for pulling lists of small planets, but parameters can be adjusted as needed.
    The default is to pull planets from the pscomppars table, in which each planet has one entry.

    Dependencies:

    astropy -- because duh everything needs astropy (but honestly only used for writing out table here)
    pyvo    -- https://pyvo.readthedocs.io/en/latest/
               PyVO lets you find and retrieve astronomical data available from archives that support standard IVOA virtual observatory service protocols:
                -Table Access Protocol (TAP) – accessing source catalogs using sql-ish queries.
                -Simple Image Access (SIA) – finding images in an archive.
                -Simple Spectral Access (SSA) – finding spectra in an archive.
                -Simple Cone Search (SCS) – for positional searching a source catalog or an observation log.
                -Simple Line Access (SLAP) – finding data about spectral lines, including their rest frequencies.

    Inputs:
    outputpath     -- Location of where to put the output data. Be careful as this will be automatically overwritten.
    sy_dist_upper  -- upper limit on the system distance (pc)
    st_rad_upper   -- upper limit on the stellar radius (solar radii)
    st_mass_upper  -- upper limit on the stellar mass (solar mass)
    pl_per_upper   -- upper limit on the planet's orbital period (days)

    * default upper limits on stellar mass and radius are M dwarf upper limits

    Returns:
    table of exoplant systems from the NASA Exoplanet Archive

    Notes:
    These parameters and constraints are most applicable to me, but this code can be borrowed or ammened to add other keys or make different cuts in the data. 

    """

    print('Accessing NASA Exoplanet Archive')
    access_url = "https://exoplanetarchive.ipac.caltech.edu/TAP"
    service = vo.dal.TAPService(access_url)
    result = service.search(f"SELECT hostname,pl_name,sy_dist,rastr,decstr,sy_vmag,sy_jmag,sy_kmag,st_mass,st_rad,st_raderr1,st_raderr2,st_teff,st_tefferr1,st_tefferr2,st_logg,st_lum,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_rade,pl_radeerr1,pl_radeerr2,pl_bmasse,pl_bmasseerr1,pl_bmasseerr2,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbsmax,pl_orbsmaxerr1,pl_orbsmaxerr2,pl_orbincl,pl_orbeccen,pl_orbeccenerr1,pl_orbeccenerr2,pl_ratror,pl_ratrorerr1,pl_ratrorerr2,pl_ratdor,pl_ratdorerr1,pl_ratdorerr2,tran_flag,rv_flag FROM pscomppars WHERE sy_dist<={sy_dist_upper} and st_rad<={st_rad_upper} and st_mass<={st_mass_upper} and pl_orbper<={pl_per_upper} ORDER BY pl_name")

    print('Table of exoplanets retrieved')
    planetSystems = result.to_table()

    print('Writing table to {}'.format(outputpath+'/NASAExoArchive_ExoplanetTable.dat'))
    ascii.write(planetSystems, outputpath+'/NASAExoArchive_ExoplanetTable.dat', overwrite=True) 

    return planetSystems

def generateExoSystem(outputpath=".", hostname="LTT 1445 A"):
    """
    This code will generate a table of transiting planets from the NASA Exoplanet Archive.
    It is written for pulling lists of small planets, but parameters can be adjusted as needed.
    The default is to pull planets from the pscomppars table, in which each planet has one entry.

    Dependencies:

    astropy -- because duh everything needs astropy (but honestly only used for writing out table here)
    pyvo    -- https://pyvo.readthedocs.io/en/latest/
               PyVO lets you find and retrieve astronomical data available from archives that support standard IVOA virtual observatory service protocols:
                -Table Access Protocol (TAP) – accessing source catalogs using sql-ish queries.
                -Simple Image Access (SIA) – finding images in an archive.
                -Simple Spectral Access (SSA) – finding spectra in an archive.
                -Simple Cone Search (SCS) – for positional searching a source catalog or an observation log.
                -Simple Line Access (SLAP) – finding data about spectral lines, including their rest frequencies.

    Inputs:
    outputpath     -- Location of where to put the output data. Be careful as this will be automatically overwritten.
    sy_dist_upper  -- upper limit on the system distance (pc)
    st_rad_upper   -- upper limit on the stellar radius (solar radii)
    st_mass_upper  -- upper limit on the stellar mass (solar mass)
    pl_rade_upper  -- upper limit on the planet radius (in earth radii)
    pl_per_upper   -- upper limit on the planet's orbital period (days)

    * default upper limits on stellar mass and radius are M dwarf upper limits
      default upper limit on planet radius is terrestrial planet upper limit

    Returns:
    table of transiting exoplant systems from the NASA Exoplanet Archive

    Notes:
    These parameters and constraints are most applicable to me, but this code can be borrowed or ammened to add other keys or make different cuts in the data. 

    """

    print('Accessing NASA Exoplanet Archive')
    access_url = "https://exoplanetarchive.ipac.caltech.edu/TAP"
    service = vo.dal.TAPService(access_url)
    result = service.search(f"SELECT hostname,pl_name,sy_dist,sy_disterr1,sy_disterr2,rastr,decstr,sy_vmag,sy_jmag,sy_kmag,st_mass,st_masserr1,st_masserr2,st_rad,st_raderr1,st_raderr2,st_teff,st_tefferr1,st_tefferr2,st_tefferr1,st_tefferr2,st_logg,st_lum,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_rade,pl_radeerr1,pl_radeerr2,pl_bmasse,pl_bmasseerr1,pl_bmasseerr2,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbsmax,pl_orbsmaxerr1,pl_orbsmaxerr2,pl_orbincl,pl_orbeccen,pl_orbeccenerr1,pl_orbeccenerr2,pl_eqt,pl_ratror,pl_ratrorerr1,pl_ratrorerr2,pl_ratdor,pl_ratdorerr1,pl_ratdorerr2,tran_flag,rv_flag FROM pscomppars WHERE hostname='{hostname}' ORDER BY pl_name")

    print('Table of exoplanets retrieved')
    planetSystem = result.to_table()

    tablename = f'NASAExoArchive_ExoplanetSystem_{hostname.replace(" ","")}'

    print('Writing table to {}'.format(outputpath+f'/{tablename}.dat'))
    ascii.write(planetSystem, outputpath+f'/{tablename}.dat', overwrite=True) 

    return planetSystem