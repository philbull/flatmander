"""
Generic class for compact sources, and a class for handling a set of compact 
sources from a catalogue.
"""
import numpy as np
import scipy.interpolate
import scipy.integrate
import astLib.astCoords
from units import *
from utils import radec_pixel_grid

class Source(object):
    
  def __init__(self, params, template_info, paramnames=None, cosmo=None):
    """
    Class to implement a source that can be added to sky maps and have its 
    amplitude/parameters sampled.
    
    Parameters
    ----------
    params : array_like
        Array of parameters of the source spatial/frequency profile.
    
    template_info : tuple(3)
        Information needed to calculate maps on the correct coordinate/pixel 
        grid. The tuple should contain:
          - template(LiteMap): Template map that defines coords/pixel grid
          - px_ra(array_like): Map of RA values for each pixel, can be None
          - px_dec(arrayLike): Map of Dec values for each pixel, can be None
        
        If the px_ra, px_dec variables are 'None', they will be precalculated. 
        This is relatively slow, so cached values of these variables should be 
        used if possible.
    
    paramnames : list_like of str, optional
        Ordered list with the names of the parameters. Can be None, in which 
        case the generic default is: ['ra', 'dec', 'width', 'beta']
    
    cosmo : Cosmology() object, optional
        Cached cosmology functions. Optional, can be None if the Source class 
        does not need to use cosmological info in its calculations.
    """
    self.cosmo = cosmo    
    self._profile = None
    self._dtheta = None
    
    # Load parameters and parameter names
    self.load_params(params, paramnames)
    
    # Get information about the map coordinates and pixel RA/Dec coords
    self.template, self.px_ra, self.px_dec = template_info
    if self.px_ra is None or self.px_dec is None:
        self.px_ra, self.px_dec = radec_pixel_grid(self.template)
  
  def load_params(self, params, paramnames):
    """
    Save parameter values and parameter names.
    """
    self.params = params
    
    # Define parameter names
    if paramnames is None:
        self.paramnames = ['ra', 'dec', 'width', 'beta']
    else:
        self.paramnames = paramnames
    
    # Save parameter values as class properties
    self.ra = self.params[self.paramnames.index('ra')]
    self.dec = self.params[self.paramnames.index('dec')]
    self.width = self.params[self.paramnames.index('width')]
    self.beta = self.params[self.paramnames.index('beta')]
  
  def spectrum(self, nu):
    """
    Spectral scaling of the source, g(nu), at a given frequency.
    
    Parameters
    ----------
    nu : float
        Frequency to evaluate spectrum at (in GHz).
    
    Returns
    -------
    g_nu : float
        Frequency scaling (dimensionless).
    """
    return (nu/140.)**self.beta
  
  def profile(self, dtheta, type=None):
    """
    Spatial profile as a fn. of distance from the center of the source.
    (Default: Gaussian with some width.)
    
    Parameters
    ----------
    dtheta : array_like of float
        Distance from center of source, dtheta, in degrees.
    
    Returns
    -------
    profile : array_like of float
        Spatial profile evaluated at a given dtheta.
    """
    return np.exp(-0.5 * (dtheta / self.width)**2.) / (2.*np.pi*self.width**2.)
  
  
  def map(self, type=None):
    """
    Spatial template (map) of the source, in uK. The frequency-dependent 
    factor is not included.
    
    Parameters
    ----------
    type : str, optional
        Which type of spatial template to return, if the source has more than 
        one type to choose from. This can be used to implement sources with 
        physically-linked emission processes (e.g. TSZ and KSZ clusters), or 
        sources with unpolarised and polarised emission.
        Default: None (returns default map for the source).
    """
    # Map of ang. distance of each pixel from centre of object
    if self._dtheta is None:
        self._dtheta = astLib.astCoords.calcAngSepDeg( self.ra, self.dec, 
                                                       self.px_ra, self.px_dec )
    dmap = self.profile(self._dtheta)
    return dmap.reshape(self.template.data.shape)
  
  def visible(self):
    """
    Determine whether the source is visible in the map or not.
    
    Returns
    -------
    visible : bool
        Whether the source is (partially) visible or not.
    """
    raise NotImplementedError()



class SourceList(object):
  
  def __init__(self, catalog, template, cosmo=None, SourceClass=Source):
    """
    Class to manage a set of sources loaded from a catalog.
    
    Parameters
    ----------
    catalog_file : str or array_like
        Path to catalog file (if string), or input catalog (if array).
    
    template : Flipper LiteMap
        Template map, used to define coordinate system and pixel grid.
    
    cosmo : Cosmology() object, optional
        If the Source class needs to calculate cosmological functions, specify 
        a Cosmology() object to provide those functions.
    
    SourceClass : class, optional
        Class that implements the source objects managed by SourceList. If none 
        is specified, it will use the generic Source() class by default.
    """
    # Flipper LiteMap template; defines the coordinate system and pixel grid
    self.template = template
    
    # Load parameters from catalogue (may be path to catalog or actual array)
    self.load_catalog(catalog)
    
    # Precompute (ra,dec) coordinates for all pixels in the map
    print " * Precomputing pixel coordinates..."
    self.px_ra, self.px_dec = radec_pixel_grid(self.template)
    self.template_info = (self.template, self.px_ra, self.px_dec)
    print "   Done."
    
    # Initialise objects for each source
    self.sources = []
    for i in range(self.Nsrc):
        src = SourceClass(params=self.catalog[i], template_info=self.template_info, 
                          cosmo=cosmo)
        self.sources.append(src)
  
  
  def load_catalog(self, catalog, simple=False):
    """
    Load parameters from catalogue. In the default mode, this reads the 
    parameter names from the header.
    
    Parameters
    ----------
    catalog : str or array_like
        Path to catalogue of sources (if string) or catalog of parameters (if 
        array_like).
    
    simple : bool, optional
        If False, this will attempt to load parameter names from the header of 
        the file. (The header line should begin with a '#' and have space-
        separated parameter names.) If True, the header is simply ignored and 
        the parameters are loaded into an array.
        Default: False.
    
    Returns
    -------
    self.catalog : array_like
        Array containing the catalogue, of shape (N_sources, N_params).
    """
    # Load catalogue from file (if filename provided)
    if isinstance(catalog, str):
        self.catalog = np.genfromtxt(catalog, skip_header=1).T
    else:
        self.catalog = catalog
    self.Nsrc = self.catalog.shape[0]
    print "\tSource catalogue: loaded %d sources" % self.Nsrc
    
    # Read parameter names from file header if requested
    self.paramnames = None
    if not simple and isinstance(catalog, str):
        f = open(catalog, 'r')
        hdr = f.readline()
        f.close()
        self.paramnames = hdr[2:-1].split(' ')
        print "\tSource catalogue params:", self.params
        
    return self.catalog
  
  def params(self):
    """
    Return a list of source parameters for all sources.
    """
    return [self.sources[i].params for i in range(self.Nsrc)]
  
  def positions(self):
    """
    Return a list of positions for all sources (RA, Dec).
    """
    return [(self.sources[i].ra, self.sources[i].dec) for i in range(self.Nsrc)]

