import numpy as np
from flipper import *
import pyfits
import astLib.astCoords
from scipy.interpolate import splrep, splev


def plot(dmap, filename, title, range=None):
    """
    Plot a data map (2D pixel grid) and save to disk.
    
    Parameters
    ----------
    dmap : array_like (2d)
        Data, in the form of a 2D array of pixel values.
    
    filename : str
        Filename for image file (file extension determines image format).
    
    title : str
        Title of the plot.
    
    range : tuple(2), optional
        Range (min, max) of data values that should be plotted.
    """
    # Plot map, trimming to value range if requested
    if range is not None:
        pylab.matshow(datamap, vmin=range[0], vmax=range[1])
    else:
        pylab.matshow(datamap)
    
    # Adjust plot properties and save to file
    pylab.colorbar()
    pylab.title('%s' % title)
    pylab.gca().set_xticks([])
    pylab.gca().set_yticks([])
    pylab.tight_layout()
    pylab.savefig(filename)
    pylab.clf()
    pylab.close()


#def generate_gaussian_window(template, pad_size, fwhm, nSigma):
def gaussian_apod_window(template, pad_size, fwhm, nsigma=5.):
    """
    Gaussian-smoothed apodisation window for use with non-periodic maps 
    (i.e. with a padding/masked region at the edges).
    
    Parameters
    ----------
    template : flipper LiteMap
        Flipper map, to be used as a template. The template is not modified.
    
    pad_size : float
        Size of padding (masked) region at edge of map (in degrees).
    
    fwhm : float
        FWHM of Gaussian apodisation at edge of mask (in arcmin).
        (argument of flipper::liteMap::liteMap::convolveWithGaussian)
    
    nsigma : float, optional
        Define Gaussian window out to this number of sigmas (default: 5).
        (argument of flipper::liteMap::liteMap::convolveWithGaussian)
    
    Returns
    -------
    window : flipper LiteMap
        Flipper map, which acts as a mask/weighting to apodise the map edges.
    """
    # Create new empty window
    window = template.copy()
    pad_size_pix = numpy.int((pad_size * numpy.pi/180.) / window.pixScaleY)
    Ny = window.Ny
    Nx = window.Nx
    
    # Mask edges of map
    window.data[:] = 1
    window.data[0:pad_size_pix,:] = 0
    window.data[:,0:pad_size_pix] = 0
    window.data[Ny-pad_size_pix:Ny,:] = 0
    window.data[:,Nx-pad_size_pix:Nx] = 0
    
    # Convolve window with Gaussian to produce apodisation window
    window = window.convolveWithGaussian(fwhm=fwhm, nSigma=nsigma)
    return window

"""
#def get_inv_noise_cov(weight, rms):
def masked_ninv(weight, rms):
    "" "
    Return masked inverse noise covariance (assuming uncorrelated noise).
    
    Parameters
    ----------
    weight : 
        xxx
    rms : 
        arcmin
    "" "
    #if numpy.min(weight.data[:]) == 0:
    #raise NotImplementedError("Zero weight not implemented; use 0 in the mask instead")
    
    RAD_TO_MIN = 180./numpy.pi * 60.
    pixArea = RAD_TO_MIN**2. * weight.pixScaleX * weight.pixScaleY
    
    sub_weight = weight.selectSubMap(30, 40, -7.3, -1)
    weight.data[:] /= (numpy.mean(sub_weight.data)*rmsArcmin**2. / pixArea)
    noiseCov = 1. / weight.data[:]
    
    #pylab.matshow(noiseCov)
    #pylab.show()
    #noiseCov /= numpy.mean(noiseCov)
    #noiseCov *= rmsArcmin**2. / pixArea

    return 1. / noiseCov
"""

#def fillWithGaussianRandomField(self, ell, Cell, bufferFactor=1):
def realize_gaussian_field(dmap, ell, Cell, bufferfactor=1):
    """
    Realize a Gaussian random field (GRF) in a map.
    
    Parameters
    ----------
    dmap : flipper LiteMap
        Empty flipper map that will be populated with realization of GRF.
        
    ell, Cell : array_like
        Multipole number and angular power spectrum for the Gaussian field to 
        be realized.
    
    bufferfactor : int, optional
        Produce a non-periodic map by realize the GRF on a larger grid than 
        the input map. The edges will then be trimmed, leaving a non-periodic 
        output map that is the same size as the input map.
        Default: 1 (periodic map).
    
    Returns
    -------
    dmap : flipper LiteMap
        Returns input LiteMap with realization of GRF.
    """
    #ft = fftTools.fftFromLiteMap(dmap)
    
    # Create empty 2D arrays with correct dimensions
    Ny = dmap.Ny * bufferfactor
    Nx = dmap.Nx * bufferfactor
    b = int(bufferfactor)
    realPart = numpy.zeros([Ny,Nx])
    imgPart  = numpy.zeros([Ny,Nx])
    
    # FFT wavenumbers
    ly = numpy.fft.fftfreq(Ny, d=dmap.pixScaleY)*(2.*numpy.pi)
    lx = numpy.fft.fftfreq(Nx, d=dmap.pixScaleX)*(2.*numpy.pi)
    
    # Create array containing the values of |l| for each Fourier-space pixel
    modLMap = numpy.zeros([Ny,Nx])
    iy, ix = numpy.mgrid[0:Ny,0:Nx]
    modLMap[iy,ix] = numpy.sqrt(ly[iy]**2. + lx[ix]**2.)
    
    # Spline the input power spectrum and evaluate on modLMap grid
    cl_spline = splrep(ell, Cell, k=3)
    ll = numpy.ravel(modLMap)
    clgrid = splev(ll, cl_spline)
    
    # Truncate C_ell evaluation at highest available wavenumber
    id = numpy.where(ll > ell.max())
    clgrid[id] = Cell[-1]
    
    # Reshape grid of Cl's and rescale by pixel area factor
    area = Nx * Ny * dmap.pixScaleX * dmap.pixScaleY
    p = numpy.reshape(clgrid, [Ny,Nx]) / area * (Nx*Ny)**2
    
    # Realize Gaussian random field with correct power spectrum
    realPart = numpy.sqrt(p) * numpy.random.randn(Ny,Nx)
    imgPart = numpy.sqrt(p) * numpy.random.randn(Ny,Nx)
    kMap = realPart + 1.j*imgPart # Fourier-space GRF realization
    
    # iFFT realization of GRF back to real-space grid and trim edges
    data = numpy.fft.ifft2(kMap)[(b-1)/2*dmap.Ny:(b+1)/2*dmap.Ny,
                                 (b-1)/2*dmap.Nx:(b+1)/2*dmap.Nx]
    dmap.data = numpy.real(dmap.data)
    return dmap


def coord_grid(template):
    """
    Get coordinates of all pixels in a template map, in ra and dec, using 
    proper coordinate-handling library (slow).
    
    Parameters
    ----------
    template : Flipper LiteMap
        Template map, used to define coordinate system and pixel grid.
    
    Returns
    -------
    ra, dec : array_like(2D)
        Pixel grids (2D arrays) of the RA and Dec values for the pixels in the 
        input template map.
    """
    Nx, Ny = template.data.shape
    idxs = np.indices((Nx, Ny))
    ra, dec = np.array( template.pixToSky( idxs[0].flatten(),
                                           idxs[1].flatten() )).T
    return ra.reshape(template.data.shape), dec.reshape(template.data.shape)


def map_from_profile(template, profile, pos, px_coords=None):
    """
    Project an angular profile onto a pixel grid.
    
    Parameters
    ----------
    template : Flipper LiteMap
        Template LiteMap, used to define coordinate system and pixel grid.
    
    profile : callable float / list of callable float
        Callable angular profile, as a function of angle from the center of the 
        profile in degrees. A list of profiles can be provided, all to be 
        evaluated on the same pixel grid with the same center coordinates.
    
    pos : tuple(2) of float
        Position of the centre of the profile in (RA, Dec).
    
    px_coords : tuple(2) of array_like, optional
        Grids of coordinates (RA, Dec) of each pixel in the template map. Can 
        be precomputed using the coord_grid() function. If None, the grids will 
        be computed on the fly (slow).
    
    Returns
    -------
    map : Flipper LiteMap / list of LiteMap
        Pixel grid with projected map of the specified profile. If a list of 
        profiles was specified for 'profile', a corresponding list of output 
        maps will be returned.
    """
    # Get position of profile centre
    ra, dec = pos
    
    # Get coordinate grid for pixels
    if px_coords is None:
        px_ra, px_dec = coord_grid(template)
    else:
        px_ra, px_dec = px_coords
    
    # Compute grid of pixel distances from profile centre
    # FIXME: Is there an approximate method that will do this faster?
    dtheta = astLib.astCoords.calcAngSepDeg(ra, dec, px_ra, px_dec)
    shp = template.data.shape # 2D map shape
    
    # Evaluate profile on this grid
    if isinstance(profile, list):
        dmap = []
        for prof in profile:
            _dmap = template.copy() # Initialise map
            _dmap.data[:] = prof(dtheta.flatten()).reshape(shp)
            dmap.append(_dmap)
    else:
        dmap = template.copy()
        dmap.data[:] = profile(dtheta.flatten()).reshape(shp)
    return dmap


#def makeEmptyCEATemplate(raSizeDeg, decSizeDeg,meanRa = 180., meanDec = 0., pixScaleXarcmin = 0.5, pixScaleYarcmin=0.5):
def empty_map(width, center=(180., 0.), pixscale=(0.5, 0.5)):
    """
    Create empty Flipper LiteMap with a given coordinate system and pixel size.
    Uses a CEA (cylindrical equal area) projection.
    
    Parameters
    ----------
    width : tuple(2) of float
        Width of the map in degrees (RA, Dec).
    
    center : tuple(2) of float, optional
        Coordinates of center of map, in degrees (RA, Dec). Default: (180, 0).
    
    pixscale : tuple(2) of float, optional
        Size of pixels in the (x, y) directions, in arcmin. Default: (0.5, 0.5).
    
    Returns
    -------
    map : Flipper LiteMap
        Empty Flipper LiteMap, with correct pixel grid and coordinate system.
    """
    # Calculate grid properties
    cdelt1 = -pixscale[0]/60.
    cdelt2 = pixscale[1]/60.
    naxis1 = numpy.int(width[0]/pixscale[0]*60. + 0.5)
    naxis2 = numpy.int(width[1]/pixscale[1]*60. + 0.5)
    refPix1 = naxis1/2.
    refPix2 = naxis2/2.
    pv2_1 = 1.0
    
    # Define properties of map as FITS CardList
    cardList = pyfits.CardList()
    cardList.append(pyfits.Card('NAXIS', 2))
    cardList.append(pyfits.Card('NAXIS1', naxis1))
    cardList.append(pyfits.Card('NAXIS2', naxis2))
    cardList.append(pyfits.Card('CTYPE1', 'RA---CEA'))
    cardList.append(pyfits.Card('CTYPE2', 'DEC--CEA'))
    cardList.append(pyfits.Card('CRVAL1', center[0]))
    cardList.append(pyfits.Card('CRVAL2', center[1]))
    cardList.append(pyfits.Card('CRPIX1', refPix1+1))
    cardList.append(pyfits.Card('CRPIX2', refPix2+1))
    cardList.append(pyfits.Card('CDELT1', cdelt1))
    cardList.append(pyfits.Card('CDELT2', cdelt2))
    cardList.append(pyfits.Card('CUNIT1', 'DEG'))
    cardList.append(pyfits.Card('CUNIT2', 'DEG'))
    hh = pyfits.Header(cards=cardList)
    wcs = astLib.astWCS.WCS(hh, mode='pyfits')
    
    # Initialize LiteMap and return
    data = numpy.zeros([naxis2,naxis1])
    ltMap = liteMap.liteMapFromDataAndWCS(data, wcs)
    return ltMap


#def writeBinnedSpectrum(lbin,clbin,fileName):
def output_binned_spectrum(l, cl, filename):
    # OBSOLETE: Should do this manually with numpy.savetxt
    raise NotImplementedError()
    
    g = open(fileName,mode="w")
    for k in xrange(len(lbin)):
        g.write("%f %e \n"%(lbin[k],clbin[k]))
    g.close()


def ninv_gaussian(template, rms):
    """
    Return uncorrelated Gaussian inverse noise covariance matrix.
    
    Parameters
    ----------
    template : Flipper LiteMap
        Template LiteMap used to set pixel grid.
    
    rms : float
        Noise RMS in uK/arcmin^2.
    
    Returns
    -------
    Ninv : Flipper LiteMap
        Inverse noise covariance matrix on pixel grid.
    """
    pix_area = RAD_TO_MIN**2. * template.pixScaleX * template.pixScaleY
    Ninv = template.copy()
    Ninv.data[:] = pixArea / rms**2. # Noise per px
    return Ninv


#def addWhiteNoise(map,rmsArcmin):
def white_noise_map(template, rms, add=True):
    """
    Create white noise map, or add white noise to an existing map.
    
    Parameters
    ----------
    dmap : Flipper LiteMap
        Input map. Either has noise added to it, or is used as a template for 
        the output noise map. A new map is returned in either case.
    
    rms : float
        Noise rms per arcmin^2.
    
    add : bool, optional
        Whether the noise should be added to the input dmap (add=True), or just 
        the noise map is returned (add=False). Default: True.
    """
    nmap = dmap.copy()
    
    if rms != 0.0:
        # Convert noise rms (per arcmin) to rms per pixel
        pixarea = (60.*180./numpy.pi)**2. * nmap.pixScaleX * nmap.pixScaleY
        rms_px = rms / numpy.sqrt(pixarea)
        
        # Realize noise map with given rms
        noise = numpy.random.normal(scale=rms_px, size=nmap.data.shape)
        if add:
            nmap.data[:] += noise[:]
        else:
            nmap.data[:] = noise[:]
    else:
        # Handle case where zero-rms nise map is requested (and not added to 
        # input dmap)
        if not add:
            nmap.data[:] = 0.
            return nmap
    return nmap
    

#def makeTemplate(m, wl, ell, maxEll, outputFile=None, print_info=False):
def beam_template(template, ell, wl, lmax, outputfile=None, debug=False):
    """
    Return 2D Fourier-space beam template (appropriate for a given pixel grid), 
    where the template is specified by harmonic coefficients w_l.
    
    Parameters
    ----------
    template : Flipper LiteMap
        Input map template (not modified by this function). The beam template 
        that is output is defined with respect to the coordinate system and 
        grid of this map.
    
    ell, wl : array_like
        Wavenumber and harmonic coefficients w_l that define beam.
    
    lmax : int
        Maximum ell value to calculate beam template out to.
    
    outputfile : string, optional
        If specified, save the resulting Fourier-space beam template to a FITS 
        file with this name.
    
    debug : bool, optional
        If True, output debug information. Default: False.
    
    Returns
    -------
    beam : Flipper LiteMap (FFT)
        Fourier-space beam template that is suitable for convolution with the 
        grid of the input map, m.
    """
    # Harmonic beam specification
    ell = numpy.array(ell)
    wl = numpy.array(wl)
    
    # Create empty Fourier-space map template
    # N.B. ell = 2pi * i / deltaX
    fT = fftTools.fftFromLiteMap(template)
    l_f = numpy.floor(fT.modLMap)
    l_c = numpy.ceil(fT.modLMap)
    fT.kMap[:,:] = 0.
    
    # Output Fourier grid info if requested
    if debug:
        print "max_lx, max_ly", fT.lx.max(), fT.ly.max()
        print "map_dx, map_dy", template.pixScaleX, template.pixScaleY
        print "map_nx, map_ny", template.Nx, template.Ny
    
    # Loop over 2D Fourier modes and assign corresponding value of w_l to each
    # FIXME: Could this be more efficient with spline evaluation?
    for i in xrange(numpy.shape(fT.kMap)[0]):
        for j in xrange(numpy.shape(fT.kMap)[1]):
            if l_f[i,j] > lmax or l_c[i,j] > lmax:
                continue
            # Do linear interpolation to get w_l for fractional l
            w_lo = wl[l_f[i,j]]
            w_hi = wl[l_c[i,j]]
            trueL = fT.modLMap[i,j]
            fT.kMap[i,j] = (w_hi - w_lo)*(trueL - l_f[i,j]) + w_lo
    
    # Copy Fourier-space beam template into new LiteMap
    beam = template.copy()
    beam.data = numpy.abs(fT.kMap)
    
    # Output beam template to file if requested
    if outputfile is not None:
        beam.writeFits(outputfile, overWrite=True)
    return beam


def bl_gaussian(width, lmax=15000):
    """
    Generate a Gaussian beam template, b_l, as a function of ell.
    
    Parameters
    ----------
    width : float
        The width of the beam, in arcmin.
    
    lmax : float, optional
        The maximum ell mode to calculate the beam templae up to. 
        Default: 15000.
    
    Returns
    -------
    b_l : array_like
        Gaussian beam spherical harmonic template.
    """
    ell = np.arange(lmax)
    return np.exp(-ell*(ell+1)*(width/RAD2MIN)**2 / 8*np.log(2) )

#def applyBeam(map, beamTemp):
def apply_beam(dmap, beam):
    """
    Convolve a datamap with a given Fourier-space beam template.
    
    Parameters
    ----------
    dmap : Flipper LiteMap / ndarray, or list of LiteMap
        Datamap to be convolved with beam template.
    
    beam : Flipper LiteMap (FFT), or list of LiteMap
        Fourier-space beam template, e.g. created by beam_template(). If a list, 
        the datamap is convolved with each beam in the list, and a list of 
        convolved maps is returned.
    
    Returns
    -------
    dmap : Flipper LiteMap / ndarray
        Input datamap convolved with beam.
    """
    # If only one beam provided, pack into list
    single = False
    if not isinstance(beam, list):
        single = True
        beam = [beam, ]
    
    # Take FFT of input data
    Fdmap = np.fft.fft2(dmap) if isinstance(dmap, np.ndarray) \
            else np.fft.fft2(dmap.data)
    
    # Loop over beams, doing beam convolution
    maps = []
    for bm in beams:
        if isinstance(dmap, numpy.ndarray):
            # Numpy array
            maps.append( np.real(numpy.fft.ifft2(bm * Fdmap)) )
        else:
            # LiteMap (need to create new instance)
            newmap = dmap.copy()
            newmap.data[:] = np.real(numpy.fft.ifft2(bm * Fdmap))
            maps.append(newmap)
    
    # Return beam-convolved map or maps
    if single return maps[0] else return maps



def power_spectrum_2d(template, l, cl):
    """
    Map an input spherical harmonic power spectrum, C_l, onto a 2D Fourier grid.
    
    Parameters
    ----------
    template : Flipper LiteMap
        Template map, used to define pixel grid that the power spectrum will be 
        projected to.
    
    l, cl : array_like
        Arrays specifying the angular power spectrum to map onto 2D Fourier 
        grid. 'cl' will be splined before evaluation.
    
    Returns
    -------
    power2d : Flipper LiteMap (FFT)
        2D Fourier-space represenation of the input angular power spectrum.
    """
    # Calculate Fourier (l_x, l_y) modes for map
    ly = np.fft.fftfreq(template.Ny, d=template.pixScaleY)*(2.*np.pi)
    lx = np.fft.fftfreq(template.Nx, d=template.pixScaleX)*(2.*np.pi)
    
    # Construct map of |l| values
    modLMap = np.zeros([template.Ny, template.Nx])
    iy, ix = np.mgrid[0:template.Ny, 0:template.Nx]
    modLMap[iy,ix] = np.sqrt(ly[iy]**2. + lx[ix]**2.)
    
    # Spline input power spectrum and evaluate on 2D grid
    s = splrep(l, cl, k=3)
    ll = numpy.ravel(modLMap)
    power2d = splev(ll, s)
    
    # Peg requested ell modes beyond lmax to C_lmax
    power2d[numpy.where(ll > np.max(l))] = cl[-1]
    
    # Rescale power spectrum by pixel scale
    power2d = np.reshape(power2d, (template.Ny, template.Nx)) \
                                     /(template.pixScaleX * template.pixScaleY)
    
    # Handle map monopole (kx=0, ky=0) mode
    power2d[0,0] = 1e-10
    
    # Return 2D power spectrum grid
    return power2d


def mask_edges(template, edge_width, apod_fwhm=None, nsigma=5.):
    """
    Mask a certain number of pixels from each edge of the map.
    
    Parameters
    ----------
    template : Flipper LiteMap
        Template datamap, used to define coordinate system and pixel grid.
    
    edge_width : int, optional
        Number of pixels to mask out from each edge of the map. Default: 0.
    
    apod_fwhm : float, optional
        FWHM of Gaussian apodisation at edge of mask (in arcmin).
        (argument of flipper::liteMap::liteMap::convolveWithGaussian)
    
    nsigma : float, optional
        Define Gaussian window out to this number of sigmas (default: 5).
        (argument of flipper::liteMap::liteMap::convolveWithGaussian)
        
    Returns
    -------
    mask : Flipper LiteMap
        Mask map containing requested edge mask.
    """
    # Create mask map from template
    mask = template.copy()
    mask.data[:] = 1
    
    # Mask a certain number of pixels around the edge of the map
    mask.data[0:edge_width,:] = 0
    mask.data[mask.Ny-edge_width:mask.Ny,:] = 0
    mask.data[:,0:edge_width] = 0
    mask.data[:,mask.Nx-edge_width:mask.Nx] = 0
    
    # Apodise mask by convolving with Gaussian
    if apod_fwhm is not None:
        mask = mask.convolveWithGaussian(fwhm=apod_fwhm, nSigma=nsigma)
    return mask


def mask_disc(template, position, radius, apod_radius=0):
    """
    Mask a disc-shaped region with a given radius.
    
    Parameters
    ----------
    template : Flipper LiteMap
        Template datamap, used to define coordinate system and pixel grid.
    
    position : tuple(2) of float
        Coordinates (RA, Dec) of centre of disc.
    
    radius : float
        Radius of disc, in arcmin.
    
    apod_radius : int, optional
        Number of pixels of apodisation to apply around disc.
        Default: 0.
        
    Returns
    -------
    mask : Flipper LiteMap
        Mask map containing requested number of holes and edge mask.
    """
    # Calculate disc size in pixels
    pixScaleArcmin = template.pixScaleX * 60.*180./numpy.pi
    radpx = numpy.int(radius / pixScaleArcmin)
    
    # Create new mask map from template
    mask = template.copy()
    mask.data[:] = 1
    
    # Find pixel at centre of disc
    ix0, iy0 = mask.skyToPix(position[0], position[1])
    
    # Loop over all pixels in entire map
    # FIXME: this could be much more efficient!
    for i in range(mask.Nx):
      for j in range(mask.Ny):
        
        # Mask pixels less than the hole radius from the centre of the hole
        rad = (i - ix0)**2 + (j - iy0)**2
        if rad < radpx**2: mask.data[j,i] = 0
        
        # Apodise mask around hole
        for pix in range(apod_radius):
            if (rad <= (radpx + pix)**2) \
              and (rad > (radpx + pix-1)**2):
                mask.data[j,i] = \
                  0.5 * (1. - numpy.cos(-numpy.pi*float(pix)/apod_radius))
    return mask


def makeMask(template, nHoles, holeSize, lenApodMask, out_pix):
    """
    Create a mask with nholes and edge mask. (OBSOLETE)
    """
    raise NotImplementedError()


def litemap_from_fits(filename, template=None):
    """
    Load image data from a FITS file into a new LiteMap, based on a given 
    LiteMap template. The image data will normally have been saved using the 
    writeFits() method of LiteMap.
    
    Parameters
    ----------
    filename : str
        Path to FITS file that data should be loaded from.
    
    template : Flipper LiteMap, optional
        LiteMap template used to define coordinate system and pixel grid. If 
        None, a new LiteMap will be constructed using info from the FITS header.
    
    Returns
    -------
    template : Flipper LiteMap
        Datamap with data from FITS file loaded into it.
    """
    # Load data from FITS file
    hdulist = pyfits.open(filename)
    hdr = hdulist[0].header
    
    # Construct new map template if one was not provided
    if template is None:
        # Construct LiteMap with properties based on FITS file header
        # N.B. raSizeDeg and decSizeDeg do not *exactly* match their values for 
        # the original LiteMap, since a float -> integer conversion was 
        # performed when the LiteMap was first saved.
        meanRa = hdr['CRVAL1']
        meanDec = hdr['CRVAL2']
        pixScaleXarcmin = -1.*60.*hdr['CDELT1']
        pixScaleYarcmin = 60.*hdr['CDELT2']
        raSizeDeg = (hdr['NAXIS1'] - 0.5) * pixScaleXarcmin / 60.
        decSizeDeg = (hdr['NAXIS2'] - 0.5) * pixScaleYarcmin / 60.
        
        # Create template using these properties
        t = empty_map( width=(raSizeDeg, decSizeDeg), center=(meanRa, meanDec), 
                       pixscale=(pixScaleXarcmin, pixScaleYarcmin) )
    else:
        # Copy input template
        t = template.copy()
    
    # Load data into LiteMap and return
    t.data[:] = hdulist[0].data
    return t

