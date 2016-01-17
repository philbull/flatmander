
import numpy as np
import astLib.astCoords
from .map_tools import *
from .units import *
import os

def galactic_to_ra_dec(self, l, b):
    """
    Return (RA, Dec) coords for given galactic coordinates (l, b).
    """
    ra, dec = astLib.astCoords.convertCoords("GALACTIC", "J2000", l, b, epoch=2000.)
    return ra, dec

def radec_pixel_grid(template):
    """
    Get coordinates of all pixels in a template map, in ra and dec, using 
    proper coordinate-handling library (slow).
    """
    Nx, Ny = template.data.shape
    idxs = np.indices((Nx, Ny))
    ra, dec = np.array( self.map_template.pixToSky( 
                          idxs[0].flatten(), idxs[1].flatten() )).T
    return ra, dec

def check_dir_exists(dirname):
    """
    Ensure that a given directory exists, and return the name of the directory.
    """
    try:
        os.makedirs(dirname)
    except:
        pass
    return dirname

def idxs_for_worker(N, rank=None, comm=None):
    """
    Return a list of indices for an array of length N, evenly divided up 
    between workers.
    """
    if comm is None:
        return range(N)
    idxs = range(rank, N, comm.size)
    return idxs

def gather_list_from_all(lst, N, comm):
    """
    Gather a list of objects created in parallel by all workers, and distribute 
    the (ordered) result to all workers.
    """
    _lst = comm.allgather(lst)
    lst = [[]]*N
    for i in range(comm.size):
        idxs = idxs_for_worker(N, i, comm)
        for j in range(len(idxs)):
            lst[idxs[j]] = _lst[i][j]
    return lst

def experiment_settings(p):
    """
    Convenience function to set-up objects that define experimental parameters, 
    such as beam, noise, and map geometry and resolution. These are needed by 
    most analysis code.
    
    Parameters
    ----------
    p: dict
        Dictionary of parameters defining input files, experimental settings etc.
    
    Returns
    -------
    template, power_2d, [beam], [ninv], [mask], [freq]
    """
    RAD2MIN = 180./np.pi * 60.
    
    # Initialize theory Cl's and beam
    l, cl_TT = np.loadtxt(p['theoryFile']).T[:2]
    cl_TT = cl_TT*2*np.pi/(l*(l+1))
    
    maketemp=p['makeTemp']
    # Generate flat-sky template
    # FIXME: Coords are fixed for now
    if maketemp['apply']:
        template = makeEmptyCEATemplate( maketemp['raSizeDeg'], maketemp['decSizeDeg'],
                                        meanRa=180., meanDec=0.,
                                        pixScaleXarcmin=maketemp['pixScaleXarcmin'],
                                        pixScaleYarcmin=maketemp['pixScaleYarcmin'] )
    print "="*50
    print "MAP PROPERTIES"
    template.info()
    print "="*50
    
    # Generate 2D power spectrum from Cl's (setting monopole to zero)
    power_2d = make2dPowerSpectrum(template, l, cl_TT)
    power_2d[0,0] = 1e-10
    
    # Get (inv.) noise, beam, and mask per band
    beams = []; ninvs = []; freqs = []
    for j in range(len(p['freq'])):
        
        # Define beam shape (assumed Gaussian for now)
        #ell, bl = np.loadtxt(p['beamFile']).T
        ell = np.arange(15000)
        bl = np.exp(-ell*(ell+1)*(p['beamSizeArcmin'][j]/RAD2MIN)**2 / 8*np.log(2) )
        # Make 2D beam template
        beamTemp = makeTemplate(template, bl, ell, np.max(l)-2, outputFile=None)
        beamTemp = beamTemp.data[:]


        # Build noise covmat
        pixArea = RAD_TO_MIN**2. * template.pixScaleX * template.pixScaleY
        InvNoiseCov = template.copy()
        InvNoiseCov.data[:] = pixArea / p['rmsArcmin'][j]**2. # Noise per px
        
        
        # Append results to lists
        beams.append(beamTemp)
        ninvs.append(InvNoiseCov.data)
        freqs.append(p['freq'][j])
    
    # Return results
    return template, power_2d, beams, ninvs, freqs


