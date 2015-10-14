"""
Class that manages a set of clusters. Loads cluster info from catalogue, 
projects cluster signal onto map, etc.
"""

import numpy as np
import scipy.interpolate
import scipy.integrate
import astLib.astCoords
import cluster_profile
from units import *

class ClusterSet(object):
  
  def __init__(self, catalogue, map_template, cosmo=DEFAULT_COSMO):
    """
    Class to manage a set of clusters. Requires filename of cluster catalogue.
    """
    # Load cosmological parameters
    self.update_cosmology(cosmo, update_clusters=False)
    
    # Get basic info about map template (for projection etc.)
    self.map_template = map_template
    
    # Load cluster info from catalogue (skips first line)
    self.catalogue = catalogue
    dat = np.genfromtxt(catalogue, skip_header=1)
    l, b, z, scal, L500, M500, R500 = np.atleast_2d(dat).T
    M500 *= 1e14 # Convert to correct units (Msun)
    
    # Trim clusters that are greater than 25*theta500 away from the map boundaries
    # TODO
    print "Visible clusters:", self.find_visible_clusters()

    # Load clusters from file and create ArnaudProfile objects for each of them
    self.clusters = []
    self.Ncl = l.size
    for i in range(self.Ncl):
        cl = cluster_profile.ArnaudProfile(params=[M500[i], R500[i], z[i]], 
                                           coords = (l[i], b[i]), cosmo=cosmo)
        self.clusters.append(cl)
    
    # Precompute (ra,dec) coordinates for all pixels in the map
    print " * Precomputing pixel coordinates..."
    self.px_ra, self.px_dec = self.get_pixel_coords()
    print "   Done."
  
  def get_profile_params(self, icl):
    """
    Return profile parameters of a given cluster.
    """
    return self.clusters[icl].get_profile_params()
  
  def get_all_profile_params(self):
    """
    Return a list of profile parameters for all clusters.
    """
    return [self.get_profile_params(i) for i in range(self.Ncl)]
  
  def get_all_positions(self):
    """
    Return a list of positions for all clusters.
    """
    return [(self.clusters[i].l, self.clusters[i].b) for i in range(self.Ncl)]
    
  def get_cluster_map(self, icl, rescale=1., maptype='tsz'):
    """
    Return TSZ/KSZ map of cluster (without frequency-dependent factor), in uK.
    The 'rescale' argument can be used to resize the cluster projected on the 
    sky, for debugging purposes.
    """
    # Convert angular scale to physical separation
    _th = astLib.astCoords.calcAngSepDeg(
               self.clusters[icl].ra, self.clusters[icl].dec,
               self.px_ra, self.px_dec )
    _r = self.da(self.clusters[icl].z) * np.tan(_th*np.pi/180.) / rescale # in Mpc
    shp = self.map_template.data.shape # 2D map shape
    
    # Get TSZ/KSZ maps
    maps = []
    if maptype == 'tsz' or maptype == 'both':
        m1 = self.clusters[icl].tsz_profile()(_r) * 1e6 # in microK
        maps.append( m1.reshape(shp) )
    if maptype == 'ksz' or maptype == 'both':
        m2 = self.clusters[icl].ksz_profile()(_r) * 1e6 # in microK
        maps.append( m2.reshape(shp) )
    return maps
  
  
  def get_cluster_map_for_profile(self, profile, ra, dec, z):
    """
    Return map of cluster (without frequency-dependent factor), in uK, for a 
    given profile and position on the sky.
    
    Parameters
    ----------
    profile: interp1d object
        Radial profile interpolation function.
    ra, dec: float
        Position on the sky.
    z: float
        Redshift of object.
    
    Returns 2D skymap in uK units.
    """
    # Convert angular scale to physical separation
    _th = astLib.astCoords.calcAngSepDeg(ra, dec, self.px_ra, self.px_dec )
    _r = self.da(z) * np.tan(_th*np.pi/180.) # in Mpc
    clmap = profile(_r)
    return clmap.reshape(self.map_template.data.shape) * 1e6 # in microK
  
  def galactic_to_ra_dec(self, l, b):
    """
    Return (RA, Dec) coords for given galactic coordinates (l, b)
    """
    ra, dec = astLib.astCoords.convertCoords("GALACTIC", "J2000", l, b, epoch=2000.)
    return ra, dec
  
  def tsz_spectrum(self, nu):
    """
    Spectral dependence of TSZ effect.
    """
    x = NU_SCALE * nu # Frequency/temperature
    g_nu = x**2. * np.exp(x) * (x/np.tanh(x/2.) - 4.) / (np.exp(x) - 1.)**2.
    return g_nu
  
  def get_pixel_coords(self):
    """
    Get coordinates of all pixels in map, in ra and dec, using proper 
    coordinate-handling library (slow).
    """
    Nx, Ny = self.map_template.data.shape
    idxs = np.indices((Nx, Ny))
    ra, dec = np.array( self.map_template.pixToSky( 
                          idxs[0].flatten(), idxs[1].flatten() )).T
    return ra, dec

  def update_cosmology(self, cosmo, update_clusters=True):
    """
    Update cosmological parameters; recompute quantities as necessary.
    """
    self.cosmo = cosmo
    
    # Get interpolating function for angular diameter distance
    _z = np.linspace(0., 2., 1000)
    integ = 1. / np.sqrt(cosmo['omega_M_0']*(1.+_z)**3. + cosmo['omega_lambda_0'])
    _da = C/(100.*cosmo['h']*(1.+_z)) * scipy.integrate.cumtrapz(integ, _z, initial=0.)
    self.da = scipy.interpolate.interp1d(_z, _da, kind='quadratic')
    
    # Update cosmology for all clusters
    if update_clusters:
      for cluster in self.clusters: cluster.set_cosmology(cosmo)
  
  def find_visible_clusters(self, max_radius=25.):
    """
    Return a list of clusters that are visible in the specified field of view.
    Clusters within max_radius*theta_500 of the field edge are classed as being 
    visible.
    """
    return
    #self.map_template
    for cl in self.clusters:
       cl.ra, cl.dec
       th500 = cl.r500 / self.da(cl.z)
       # FIXME
       

"""
def interp_pixel_grid(ix, iy, Nx, Ny, points):
    " ""
    Bilinear interpolation on pixel grid, using algorithm from 
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    Expects points = f(0,0), f(xmax, 0), f(0, ymax), f(xmax, ymax)
    ix, iy are pixel IDs
    Nx, ny are the x,y dimensions of the grid, in pixels
    " ""
    q11, q21, q12, q22 = points
    val = ( q11 * (Nx-1 - ix) * (Ny-1 - iy) +
            q21 * (ix) * (Ny-1 - iy) +
            q12 * (Nx-1 - ix) * (iy) +
            q22 * (ix) * (iy)
           ) / ((Nx-1) * (Ny-1) + 0.0)
    return val

def get_pixel_coords_fast(lmap):
    " ""
    Get coordinates of all pixels in map, in ra and dec, using fast 
    interpolation approximation (probably fails at poles).
    " ""
    # Get edge coordinates
    Nx, Ny = lmap.data.shape
    edges = [ lmap.pixToSky(0, 0),    lmap.pixToSky(Nx-1, 0), 
              lmap.pixToSky(0, Ny-1), lmap.pixToSky(Nx-1, Ny-1) ]
    edges = np.array(edges).T
    
    # Interpolate over entire pixel grid
    idxs = np.indices((Nx, Ny))
    ix = idxs[0].flatten(); iy = idxs[1].flatten()
    ra = interp_pixel_grid(ix, iy, Nx, Ny, edges[0])
    dec = interp_pixel_grid(ix, iy, Nx, Ny, edges[1])
    return ra, dec
"""
