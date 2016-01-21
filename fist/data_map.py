
class DataMap(object):

  def __init__(self, freq, datamap, beam, Ninv, mask):
    """
    Class describing a datamap and associated information, including beam, 
    inverse noise covariance, mask, and frequency.
    
    Parameters
    ----------
    freq : float
        Centre frequency for this frequency channel, in GHz.
    
    datamap : str or Flipper LiteMap
        Data for this frequency channel on a pixel grid.
        If str, will be interpreted as a FITS filename to load data from.
        If LiteMap, will be used directly.
        (Also used to set coordinate system of the pixel grid.)
    
    beam : Flipper LiteMap (FFT)
        2D Fourier-space beam template for this frequency channel.
    
    Ninv : Flipper LiteMap
        Map of inverse noise covariance per pixel.
    
    mask : Flipper LiteMap
        Map of masked/unmasked pixels.
    """
    
    # Set frequency info
    self.freq = freq
    
    # Load datamap
    self.update_datamap(datamap)
    
    # Load beam
    if isinstance(beam, float):
        self.update_beam(width=beam)
    elif isinstance(beam, str):
        self.update_beam(beamfile=beam)
    elif isinstance(beam, np.ndarray):
        self.update_beam(bl=beam)
    else:
        self.update_beam()
    
    # Load inverse noise covmat
    if isinstance(Ninv, float):
        self.update_ninv(rms=Ninv)
    elif isinstance(Ninv, str):
        self.update_ninv(noisefile=Ninv)
    elif isinstance(Ninv, np.ndarray) or isinstance(Ninv, type(self.datamap)):
        self.update_ninv(Ninv=Ninv)
    else:
        self.update_ninv()
    
    # Load mask
    self.update_mask(mask)
  
  
  def update_datamap(self, datamap):
    """
    Load datamap from file, or set using passed argument 'datamap'.
    """
    if isinstance(datamap, str):
        self.datamap = litemap_from_fits(datamap)
    else:
        self.datamap = datamap
    try:
        self.datamap.data
    except:
        raise TypeError("datamap must be Flipper LiteMap.")


  def update_beam(self, width=None, beamfile=None, bl=None, lmax=15000):
    """
    Update Fourier-space beam template.
    
    Parameters
    ----------
    width : float, optional
        If specified, creates a Gaussian beam with this width (in arcmin).
    
    beamfile : str, optional
        If specified, loads beam template from a file.
    
    bl : array_like, optional
        If specified, 
    
    lmax : float, optional
        Max. SH ell mode to calculate the beam template up to. Default: 15000.
    """
    if width is not None:
        # Create Gaussian beam template
        ell = np.arange(lmax)
        _bl = bl_gaussian(width, lmax)
        self.beam = beam_template(self.datamap, ell, _bl, lmax-2)
        
    elif beamfile is not None:
        # Load beam template directly from file
        raise NotImplementedError('Loading from beamfile not yet implemented.')
    
    elif bl is not None:
        # Create beam template with user-specified angular profile, b_l
        ell = np.arange(lmax)
        self.beam = beam_template(self.datamap, ell, bl, lmax-2)
    else:
        raise ValueError('Must specify one of: width, beamfile, bl')
  
  
  def update_ninv(self, Ninv=None, noisefile=None, rms=None):
    """
    Construct inverse noise covariance map for this datamap.
    
    Parameters
    ----------
    Ninv : ndarray or Flipper LiteMap, optional
        If specified, use this array/LiteMap as N_inv directly.
    
    noisefile : str, optional
        If specified, load N_inv from this file. (Not Implemented)
    
    rms : float, optional
        If specified, construct an isotropic Gaussian N_inv with this rms 
        (in uK/arcmin^2).
    """
    if Ninv is not None:
        # Add Ninv directly
        if isinstance(Ninv, np.ndarray):
            # If ndarray, pack into LiteMap
            self.Ninv = self.datamap.copy()
            self.Ninv.data[:] = Ninv
        else:
            self.Ninv = Ninv
            try:
                self.Ninv.data
            except:
                raise TypeError('Ninv argument must be LiteMap or ndarray')
    
    elif noisefile is not None:
        # Load N_inv directly from file
        raise NotImplementedError('Loading from noisefile not yet implemented.')
    
    elif rms is not None:
        # Create Gaussian Ninv with given noise rms
        self.Ninv = ninv_gaussian(self.datamap, rms)
    
    else:
        raise ValueError('Must specify one of: Ninv, noisefile, rms')


  def update_mask(self, mask=None, maskfile=None):
    """
    Specify a mask for the datamap. If no arguments are passed, a blank mask 
    will be used.
    
    Parameters
    ----------
    mask : ndarray or Flipper LiteMap, optional
        If specified, use this array/LiteMap as the mask directly.
    
    maskfile : str, optional
        If specified, load the mask from this FITS file.
    """
    if mask is not None:
        # Add mask directly
        if isinstance(mask, np.ndarray):
            # If ndarray, pack into LiteMap
            self.mask = self.datamap.copy()
            self.mask.data[:] = mask
        else:
            self.mask = mask
            try:
                self.mask.data
            except:
                raise TypeError('mask argument must be LiteMap or ndarray')
            
    elif maskfile is not None:
        # Load mask from FITS file
        self.mask = litemap_from_fits(maskfile)
        
    else:
        # No mask specified; set to 'no mask' (i.e. unity everywhere)
        self.mask = self.datamap.copy()
        self.mask.data[:] = 1.
