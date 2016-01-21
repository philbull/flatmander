"""
Generic class for compact sources, and a class for handling a set of compact 
sources from a catalogue.
"""
import numpy as np
import scipy.interpolate
import scipy.integrate
import astLib.astCoords
from units import *

class Fluctuations(object):
    
  def __init__(self, powspec, cosmo=None):
    """
    Class to implement a component that is a temperature fluctuation field with 
    some angular power spectrum (e.g. the CMB).
    
    Parameters
    ----------
    powspec : Flipper LiteMap (FFT)
        LiteMap containing theoretical/a priori Fourier-space power spectrum 
        for the fluctuation field; to be used as a prior (i.e. not just the 
        measured power spectrum of the field.)
    
    cosmo : Cosmology() object, optional
        Cached cosmology functions. Optional, can be None if the Source class 
        does not need to use cosmological info in its calculations.
    """
    
    # Set position
    self.update_powspec(powspec)
    
    # Get information about the map coordinates and pixel RA/Dec coords
    self.template = self.powspec
  
  
  def update_powspec(self, powspec):
    """
    Update 2D power spectrum.
    
    Parameters
    ----------
    powspec : Flipper LiteMap (FFT)
        Theoretical 2D Fourier-space power spectrum, to be used as a prior.
    """
    self.powspec2d = powspec
  
  
  def update_params(self, params, paramnames=None):
    """
    Save parameter values and parameter names.
    """
    raise NotImplementedError()
  
  
  def spectrum(self, nu, type=None):
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
    return np.ones_like(nu)



class CMB(Fluctuations):
    
  def __init__(self, powspec, cosmo=None):
    """
    Class to implement a CMB temperature fluctuation component, constrained by 
    some power spectrum.
    """
    # Initialise parent class
    super(CMB, self).__init__(powspec=powspec, cosmo=cosmo)



class PowerlawForeground(Fluctuations):
    
  def __init__(self, params, powspec, paramnames=None, cosmo=None):
    """
    Class to implement a foreground temperature fluctuation component, with an 
    overall powerlaw frequency scaling, and constrained by some power spectrum.
    """
    raise NotImplementedError()
    
    # Initialise parent class
    super(PowerlawForeground, self).__init__(powspec=powspec, cosmo=cosmo)

