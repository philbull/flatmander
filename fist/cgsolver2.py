import numpy as np
import fft
from fluctuations import *
from compact_source import *
from map_tools import map_from_profile


class LinearSystem(object):
  def __init__(self, datamaps, components, types):
    """
    Class describing the structure of the linear system used for the 
    constrained realisation step.
    
    Parameters
    ----------
    datamaps : list of DataMap
        List of DataMap objects, each of which describes the data and 
        instrumental properties for a given frequency channel.
    
    components : list of Fluctuations/Source/SourceList
        List of objects describing components of the sky model. These will be 
        used to provide information for defining the linear operator matrix.
    
    types : list of str
        If components in the list have different possible types, specify the 
        type of the component here (e.g. 'tsz' or 'ksz').
    """
    # Set properties
    self.datamaps = datamaps
    self.components = components
    self.types = types
    self.nbands = len(self.datamaps)
    
    # Sanity checks
    # (1) Check for zero-length data and components
    if len(self.datamaps) < 1: raise ValueError("Requires at least one datamap.")
    if len(self.components) < 1: raise ValueError("Requires at least one sky component.")
    
    # (2) Check that components are of usable types
    for c in self.components:
        if not isinstance(c, Fluctuations) or \
           not isinstance(c, SourceList) or \
           not isinstance(c, Source):
        raise TypeError("Component", c, "is not of recognised type.")
    
    # Define a map template (for coord system/pixel grid); just use the first 
    # input datamap for this
    self.template = self.datamaps[0].datamap
    

  def construct_source_maps(self, px_coords=None, verbose=True):
    """
    Construct pixel maps for all defined sources and collections of sources.
    
    Parameters
    ----------
    px_coords : tuple(2) of Flipper LiteMap, optional
        Maps of the (RA, Dec) coordinates of the pixels in the pixel grid, 
        produced by the coord_grid() function. If not specified, an appropriate 
        grid will be computed on the fly (slow).
    
    verbose : bool, optional
        Whether to output progress information. Default: True.
    """
    # Generate pixel coordinate grid if a precomputed one was not provided
    if px_coords is not None:
        px_coords = coord_grid(template)
    
    # Get list of beams for each band
    beams = [d.beam for d in self.datamaps]
    
    # Loop over all components, find Source/SourceList components, and construct 
    # maps from their profiles.
    nmaps = 0
    self.source_maps = []
    for c, t in zip(self.components, self.types):
        if isinstance(c, Source):
            # Component is a single source
            prof = c.profile(type=t) if t is not None else c.profile()
            m = map_from_profile(self.template, prof, pos=(c.ra, c.dec), 
                                 px_coords=px_coords)
            
            # Loop over bands, doing beam convolution
            self.source_maps.append( apply_beam(m, beams) )
            nmaps += 1
            if verbose: print "\t construct_source_maps() finished map %d" % nmaps
            
        elif isinstance(c, SourceList):
            # Component is a collection of sources
            beams = [d.beam for d in self.datamaps]
            srcmaps = []
            for src in c.sources:
                prof = src.profile(type=t) if t is not None else src.profile()
                m = map_from_profile(self.template, prof, pos=(src.ra, src.dec), 
                                     px_coords=px_coords)
                
                # Loop over bands, doing beam convolution
                srcmaps.append( apply_beam(m, beams) )
                nmaps += 1
                if verbose: print "\t construct_source_maps() finished map %d" % nmaps
            self.source_maps.append(srcmaps)
            
        else:
            # Not a compact source component; set to None
            self.source_maps.append(None)
    
    # Output progress information if requested
    if verbose: print "\t construct_source_maps() finished all maps."


  def construct_precond(self, precond_rms, precond_type='default'):
    """
    Construct a simple preconditioner for each Fluctuations component in the 
    linear system. Sets self.precond2d, which is a list of preconditioners, one 
    for each component in self.components.
    
    Parameters
    ----------
    precond_rms : float
        RMS noise used to define scaling of preconditioner.
    
    precond_type : str, optional
        Select which preconditioner to use. At the moment, only one is 
        supported. Default: 'default'.
    """
    if precond_type is not 'default':
        raise NotImplementedError("precond_type '%s' is not supported." \
                                  % precond_type)
    
    # Just use the beam of the first band as the effective beam
    beam = self.datamaps[0].beam
    
    # Effective noise of preconditioner
    nl_TT = precond_rms**2.
    
    # Loop through all components and construct preconditioners for Fluctuations 
    # components only
    self.precond2d = []
    for c in self.components:
        if isinstance(c, Fluctuations):
            p1 = np.sqrt(c.powspec2d * nl_TT / (nl_TT + c.powspec2d * beam**2.))
            self.precond2d.append(p1)
        else:
            self.precond2d.append(None)
    

  def compute_x0(self):
    """
    Compute an initial guess for the linear system solution. Currently just 
    set to zero for all components.
    
    Returns
    -------
    x0 : list of array_like
        Initial amplitude parameter arrays for each component.
    """
    # Count total number of amplitude parameters for all components
    x0 = []
    for c in self.components:
        if isinstance(c, Fluctuations):
            x0.append( np.zeros(self.template.data.size) )
        elif isinstance(c, Source):
            x0.append( np.zeros(1) )
        elif isinstance(c, SourceList):
            x0.append( np.zeros(len(c.sources)) )
        else:
            raise TypeError("Component type '%s' is not supported." % type(c))
    return x0
    
  
  def compute_rhs(self):
    """
    Compute right-hand side of linear system Ax = b.
    """
    nx = self.template.data.nx
    ny = self.template.data.ny
    
    def rhs_fluctuations(self, precond2d):
        """
        Calculate RHS of linear system for a fluctuation component: 
            y = S^1/2 B N^-1 d
        where S is (preconditioned) signal covariance matrix (Cl's).
        
        Parameters
        ----------
        precond2d : Flipper LiteMap
            Precomputed preconditioner for this component.
        
        Returns
        -------
        y : array_like
            RHS of linear system for this component.
        """
        # Note: Reshaping operations are required to go between 2D pixel arrays 
        # and 1D amplitude vector (for linear system)
        
        # Loop over frequency bands to compute RHS
        d2 = 0
        for d in self.datamaps:
            d1 = d.Ninv.data.reshape((ny,nx)) * d.datamap.data.reshape((ny,nx))
            a_l = precond2d * d.beam.data.reshape((ny,nx)) \
                * fft.fft(d1, axes=[-2,-1]) 
            d1 = np.real(fft.ifft(a_l, axes=[-2,-1], normalize=True))
            d1 = np.reshape(d1, (nx*ny)) # Reshape to 1D amplitude vector
            d2 += d1
        return d2
      
    def rhs_source(self, component, source_maps, type=None):
        """
        Calculate RHS of linear system for a compact source (Source) or 
        collection of sources (SourceList):
            y = F^T B^T N^-1 d
        where F is the spatial template of each source.
        
        Parameters
        ----------
        component : Source or SourceList instance
            Class containing info about this component; either a Source or 
            SourceList (or a class derived from one of these).
            
        source_maps : list
            List of beam-convolved source maps for each source, for each band.
            If component is a Source, this is a 1D list (for each band).
            If component is a SourceList, this is a 2D list (for each source, 
            for each band).
        
        type : str, optional
            If the component has different types, specify which type (e.g. 
            'tsz' or 'ksz').
        
        Returns
        -------
        y : array_like
            RHS of linear system for this component.
        """
        # If it's just a single source, insert into a list so we can loop
        if isinstance(component, Source):
            sources = [component,]
            source_maps = [source_maps,]
            Nsrc = 1
        else:
            sources = component.sources
            Nsrc = component.Nsrc
        
        # Loop over sources, computing RHS
        d2 = np.zeros(Nsrc)
        for i in range(Nsrc):
            for j in range(self.nbands):
                d = self.datamaps[j]
                
                # Get frequency scaling
                g_nu = sources[i].spectrum(d.freq, type=type) if type is not None \
                       else sources[i].spectrum(d.freq)
                
                # FIXME: Do we *need* to reshape?
                d1 = d.Ninv.data.reshape((ny,nx)) * d.datamap.data.reshape((ny,nx))
                d2[i] += np.sum(d1 * source_maps[i][j] * g_nu)
        return d2
    
    def rhs_monopole(self):
        """
        Calculate RHS of linear system for a monopole term.
        """
        d1 = 0
        for d in self.datamaps:
            d1 += np.sum(  d.datamap.data.reshape((ny,nx)) \
                         * d.Ninv.data.reshape((ny,nx)) )
        return d1
    
    # Per-band white noise realisations
    bnu = [np.random.randn(ny,nx) * d.Ninv.data**0.5 for d in self.datamaps]
    
    # Loop over components, constructing RHS for each
    b = []
    for c, precond, src_maps, type in zip(self.components, self.precond2d, 
                                          self.source_maps, self.types):
        # Fluctuations component
        if isinstance(c, Fluctuations):
            # Get Wiener-filtered RHS
            y = self.rhs_fluctuations(precond)
            
            # First white noise realisation; convolve white noise map with beam 
            # and multiply by signal covmat S^1/2 in harmonic space
            b1 = np.random.randn(ny,nx)
            a_l = fft.fft(b1, axes=[-2,-1]) * precond * c.powspec2d**-0.5
            b1 = np.real(fft.ifft(a_l, axes=[-2,-1], normalize=True))
            
            # Apply beam and precond. to per-band white noise realisations
            b2 = 0
            for _bnu, d in zip(bnu, self.datamaps):
                a_l = fft.fft(_bnu, axes=[-2,-1]) * d.beam.data * precond
                b2 += np.real(fft.ifft(a_l, axes=[-2,-1], normalize=True))
            
            # Construct RHS for Fluctuations component
            b.append( y + b1.reshape((nx*ny)) + b2.reshape((nx*ny)) )
        
        # SourceList component
        elif isinstance(c, SourceList):
            # Get Wiener-filtered RHS
            y = self.rhs_source(c, src_maps, type=type)
            
            # Noise realisation for each source in SourceList
            b1 = np.zeros(c.Nsrc)
            for i in range(c.Nsrc):
                src = c.sources[i]
                
                # Loop over bands
                for j in range(len(self.datamaps)):
                    d = self.datamaps[j]
                    
                    # Get frequency scaling
                    g_nu = src.spectrum(d.freq, type=type) if type is not None \
                           else src.spectrum(d.freq)
                    
                    # Add to noise realisation part of RHS
                    b1[i] += np.sum( bnu[j] * g_nu * src_maps[i][j] )
            
            # Construct RHS for SourceList component
            b.append( y + b1 )
        
        # Source component
        elif isinstance(c, Source):
            # Get Wiener-filtered RHS
            y = self.rhs_source(c, src_maps, type=type)
            
            # Noise realisation for the source; loop over bands
            b1 = 0
            for j in range(len(self.datamaps)):
                d = self.datamaps[j]
                
                # Get frequency scaling
                g_nu = src.spectrum(d.freq, type=type) if type is not None \
                       else src.spectrum(d.freq)
                
                # Add to noise realisation part of RHS
                b1 += np.sum( bnu[j] * g_nu * src_maps[j] )
            
            # Construct RHS for Source component
            b.append( y + b1 )
        
        # Monopole component
        elif isinstance(c, Monopole):
            # Get Wiener-filtered RHS
            y = self.rhs_monopole()
            
            # Sum monopole over per-band white noise realisations
            b1 = 0
            for _b in bnu: b1 += np.sum(_b)
            
            # Construct RHS for Monopole component
            b.append( y + b1 )
        
        # Unrecognised
        else:
            raise TypeError("Component %s not of recognised type." % c)
    
    # Return RHS list
    return b
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
  def apply_mat(self, x):
    """
    Apply the full block linear matrix to some vector x.
    
    Parameters
    ----------
    x : array_like
        Vector of amplitudes to which the linear operator will be applied.
    """
    
    # (Fluct x Fluct)
    # Apply (S^-1 + A N^-1 A) x
    
    
    #Apply the CMB x (Monopole + TSZ + KSZ) terms in one block:
    #        (A^T N^-1 A T) x_mono
    #        (A^T N^-1 A F) x_tsz
    #        (A^T N^-1 A K) x_ksz
    
    #        (T^T A^T N^-1 A) x_cmb
    #        (F^T A^T N^-1 A) x_cmb
    #        (K^T A^T N^-1 A) x_cmb
    
    
    
    
  def applyMat(my_map, linsys_setup):
    """
    Multiply my_map by the matrix operator, defined in Eq. 19 of Eriksen 
    et al. (2008).
    """
    
    #datamaps, ninvs, beams, freqs, power_2d, precond_2d, clumaps, g_nu, \
    #                                       map_prop = linsys_setup
    
    nx = self.template.nx
    ny = self.template.ny
    pixScaleX = self.template.pixScaleX
    pixScaleY = self.template.pixScaleY
    nfreq = len(self.datamaps)
    
    #nFreq = len(g_nu); nCluster = len(clumaps[0])

    # Always apply beam * precond
    beam_prec=[]
    for f in range(nFreq):
        beam_prec+=[beams[f][:,:ny/2+1]*precond_2d[:,:ny/2+1]]
    precond_2d=precond_2d[:,:ny/2+1]
    power_2d=power_2d[:,:ny/2+1]
    
    ksz = False
    if len(clumaps) == 2: ksz = True
    
    # Routines to perform block matrix multiplication defined in Eriksen Eq. 19
    
    def apply_cmb_cmb(d0):
        """
        Apply (S^-1 + A N^-1 A) x
        """
        d1 = d0.copy()
        d1 = numpy.reshape(d1,(nx,ny))
        a_l = fft.rfft(d1,axes=[-2,-1])
        
        c_l = 0
        for f in range(nFreq):

            b_l = a_l * beam_prec[f]
            d2 = fft.irfft(b_l,axes=[-2,-1],normalize=True)
            d2 *= ninvs[f]
            b_l = fft.rfft(d2,axes=[-2,-1])
            c_l += b_l * beam_prec[f]
                
        d2 = fft.irfft(c_l,axes=[-2,-1],normalize=True)
        d1 = fft.irfft(precond_2d**2 * a_l/power_2d,axes=[-2,-1],normalize=True)
        
        d2 += d1
        
        return d2.reshape((nx*ny,))
    
    def apply_cmb_foreground_block(dc, dm, dt, dk=None):
        """
        Apply the CMB x (Monopole + TSZ + KSZ) terms in one block:
            (A^T N^-1 A T) x_mono
            (A^T N^-1 A F) x_tsz
            (A^T N^-1 A K) x_ksz
            
            (T^T A^T N^-1 A) x_cmb
            (F^T A^T N^-1 A) x_cmb
            (K^T A^T N^-1 A) x_cmb
        """
        ksz = False
        if dk is not None: ksz = True
        
        # (A^T N^-1 A T) x_mono; (A^T N^-1 A F) x_tsz; (A^T N^-1 A K) x_ksz
        b_lt = 0; b_lk = 0; b_lm = 0
        for f in range(nFreq):
          mct = 0; mck = 0
          for ic in range(nCluster):
            mct += dt[ic] * ninvs[f] * clumaps[0][ic][f] * g_nu[f]
            if ksz: mck += dk[ic] * ninvs[f] * clumaps[1][ic][f]
        
          b_lm += fft.rfft(dm * ninvs[f],axes=[-2,-1])  * beam_prec[f]
          b_lt += fft.rfft(mct,axes=[-2,-1]) * beam_prec[f]
          if ksz: b_lk += fft.rfft(mck,axes=[-2,-1]) * beam_prec[f]

        mcm = fft.irfft(b_lm,axes=[-2,-1],normalize=True).reshape((nx*ny,))
        mct = fft.irfft(b_lt,axes=[-2,-1],normalize=True).reshape((nx*ny,))
        if ksz: mck = fft.irfft(b_lk,axes=[-2,-1],normalize=True).reshape((nx*ny,))
        
        # (T^T A^T N^-1 A) x_cmb; (F^T A^T N^-1 A) x_cmb; (K^T A^T N^-1 A) x_cmb
        mc = dc.copy().reshape((nx,ny))
        a_l = fft.rfft(mc,axes=[-2,-1])
        mtc = numpy.zeros(nCluster)
        mkc = numpy.zeros(nCluster)
        mmc = 0
        for f in range(nFreq):
          b_l = a_l * beam_prec[f]
          mc = fft.irfft(b_l,axes=[-2,-1],normalize=True)
          mmc += numpy.sum(mc * ninvs[f])
          for ic in range(nCluster):
            mtc[ic] += numpy.sum(mc * ninvs[f] * clumaps[0][ic][f] * g_nu[f])
            if ksz: mkc[ic] += numpy.sum(mc * ninvs[f] * clumaps[1][ic][f])
        
        if ksz: return mct, mcm, mck, mtc, mmc, mkc
        return mct, mcm, mtc, mmc
    
    
    def apply_foreground_block(m0, t0, k0=None):
        """
        Apply the TSZ + KSZ + Monopole terms in one block:
            [ (T^T A^T N^-1 A F)  (T^T A^T N^-1 A K)  (T^T A^T N^-1 A T) ] (x_mono)
            [ (F^T A^T N^-1 A F)  (F^T A^T N^-1 A K)  (F^T A^T N^-1 A T) ] (x_tsz)
            [ (K^T A^T N^-1 A F)  (K^T A^T N^-1 A K)  (K^T A^T N^-1 A T) ] (x_ksz)
        """
        ksz = True if k0 is not None else False
        
        dtt, dkk, dtk, dkt = [numpy.zeros(nCluster) for i in range(4)]
        mtt, mkk, mtk, mkt = [numpy.zeros((nCluster, nCluster)) for i in range(4)]
        dmm, dmk, dmt = [0 for i in range(3)]
        dkm, dtm = [numpy.zeros(nCluster) for i in range(2)]
        
        # TODO: This could probably be more efficient (e.g. using np.outer)
        for f in range(nFreq):
          dmm += numpy.sum(ninvs[f]) * m0
          
          # Loop through clusters
          for ic in range(nCluster):
            dmt += numpy.sum( ninvs[f] * g_nu[f] * clumaps[0][ic][f] * t0[ic] )
            dtm[ic] += numpy.sum( ninvs[f] * g_nu[f] * clumaps[0][ic][f] * m0 )
            if ksz:
              dmk += numpy.sum( ninvs[f]  * clumaps[1][ic][f] * k0[ic] )
              dkm[ic] += numpy.sum( ninvs[f]  * clumaps[1][ic][f] * m0 )
            
            for jc in range(0, ic+1):
              mtt[ic,jc] = numpy.sum(  ninvs[f] * g_nu[f]**2. \
                                     * clumaps[0][ic][f] * clumaps[0][jc][f] )
              if ksz:
                mkk[ic,jc] = numpy.sum(  ninvs[f] \
                                       * clumaps[1][ic][f] * clumaps[1][jc][f] )
                mtk[ic,jc] = numpy.sum(  ninvs[f] * g_nu[f] \
                                       * clumaps[0][ic][f] * clumaps[1][jc][f] )
                mkt[ic,jc] = numpy.sum(  ninvs[f] * g_nu[f] \
                                       * clumaps[1][ic][f] * clumaps[0][jc][f] )
              # Mirror indices
              mtt[jc,ic] = mtt[ic,jc]
              if ksz:
                mkk[jc,ic] = mkk[ic,jc]
                mtk[jc,ic] = mtk[ic,jc]
                mkt[jc,ic] = mkt[ic,jc]
          
          # Add total contribs. for this band
          dtt += numpy.dot(mtt, t0)
          if ksz:
            dkk += numpy.dot(mkk, k0)
            dtk += numpy.dot(mtk, k0)
            dkt += numpy.dot(mkt, t0)
            
        if ksz: return dtt, dkk, dmm, dtk, dkt, dmk, dkm, dtm, dmt
        return dtt, dmm, dtm, dmt
    
    # Apply block matrix multiplications and return
    # FIXME: What if KSZ not used?
    x0 = my_map[:nx*ny]
    x1 = my_map[nx*ny:nx*ny+1]
    x2 = my_map[nx*ny+1:nx*ny+nCluster+1]
    if ksz: x3 = my_map[nx*ny+nCluster+1:nx*ny+2*nCluster+1]
    
    # Multiply input vector in blocks
    #t=time.time()
    dcc = apply_cmb_cmb(x0)
    #print 'CMB', time.time()-t
    if ksz:
        dct, dcm, dck, dtc, dmc, dkc = apply_cmb_foreground_block(x0, x1, x2, x3)
        dtt, dkk, dmm, dtk, dkt, dmk, dkm, dtm, dmt = apply_foreground_block(x1, x2, x3)
        x_new_0 = dcc + dct + dck + dcm
        x_new_1=  dmc + dmt + dmk + dmm
        x_new_2 = dtc + dtt + dtk + dtm
        x_new_3 = dkc + dkt + dkk + dkm
        x_new = numpy.concatenate((x_new_0, x_new_1, x_new_2, x_new_3))
    else:
        #t=time.time()
        dct, dcm, dtc, dmc = apply_cmb_foreground_block(x0, x1, x2)
        #print 'CMB-F', time.time()-t
        
        #t=time.time()
        dtt, dmm, dtm, dmt = apply_foreground_block(x1, x2)
        #print 'F', time.time()-t
        
        x_new_0 = dcc + dct + dcm
        x_new_1 = dmc + dmt + dmm
        x_new_2 = dtc + dtt + dtm
        x_new = numpy.concatenate((x_new_0, x_new_1, x_new_2))

    return x_new
    
    
    
    

