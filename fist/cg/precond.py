
from flipper import *
from ..gibbs_tools import *
import fft

def prep_system( expt_setup, power_2d, cluster_set, precond_rms, vcov=None ):
    """
    Precomputes all of the quantities needed to solve the linear system, 
    including a preconditioner and the cluster spatial templates.
    
    Parameters
    ----------
    
    expt_setup : tuple
        Tuple of lists of quantities that define the experimental setup. See 
        utils.experiment_settings() for the definitions; the order is:
        expt_setup = (datamaps, ninvs, beams, freqs)
    
    power_2d : array_like
        CMB power spectrum.
    
    cluster_set : ClusterSet object
        Set of clusters from a catalogue.
    
    precond_rms : float
        The reference RMS noise level for the preconditioner, in uK/arcmin^2 (?). 
        Will usually be the RMS for the lowest band of the experiment.
    
    vcov : array_like (2D), optional
        KSZ peculiar velocity covariance matrix of shape (nCluster, nCluster). 
        If left unspecified, only TSZ amplitudes will be sampled.
    
    reuse : list, optional
        
    """
    datamaps, ninvs, beams, freqs = expt_setup
    
    # Define map and cluster catalogue properties
    nx, ny = (datamaps[0].Nx, datamaps[0].Ny)
    pixScaleX, pixScaleY = (datamaps[0].pixScaleX, datamaps[0].pixScaleY)
    map_prop = (nx, ny, pixScaleX, pixScaleY)
    
    nFreq = len(freqs)
    nCluster = cluster_set.Ncl
    
    # Define CG preconditioner
    nl_TT = precond_rms**2.
    precond_2d = numpy.sqrt(power_2d * nl_TT / (nl_TT + power_2d * beams[0]**2))
    
    # Get TSZ freq. scaling
    # TODO: Parallelise get_cluster_map()
    g_nu = [cluster_set.tsz_spectrum(f) for f in freqs]

    # Build KSZ cluster maps and invert vel. covariance matrix (if KSZ is enabled)
    use_ksz = False; ivcov = None
    if vcov is not None:
        use_ksz = True
        ivcov = build_vcov_inverse(vcov)
    
    # Build (freq.-dep.) beam-convolved maps of TSZ/KSZ effect for each cluster
    maps = build_cluster_maps(cluster_set, beams, use_ksz=use_ksz)
    
    # Construct tuple containing all useful settings/quantities
    linsys_setup = ( datamaps, ninvs, beams, freqs, power_2d, precond_2d, 
                     maps, g_nu, map_prop, vcov, ivcov )
    return linsys_setup

def update_system_setup(linsys_setup, vcov=None, ivcov=None, maps=None):
    """
    Re-build the system setup tuple created by prep_system() after some of its 
    elements have been updated.
    """
    datamaps, ninvs, beams, freqs, power_2d, precond_2d, _maps, g_nu, \
      map_prop, _vcov, _ivcov = linsys_setup
    
    # Update relevant components
    if vcov is not None: _vcov = vcov
    if ivcov is not None: _ivcov = ivcov
    if maps is not None: _maps = maps
    
    # Construct updated linear system setup
    new_linsys_setup = ( datamaps, ninvs, beams, freqs, power_2d, precond_2d, 
                         _maps, g_nu, map_prop, _vcov, _ivcov )
    return new_linsys_setup

def build_cluster_maps(cluster_set, beams, use_ksz=False):
    """
    Build maps of TSZ (and optionally KSZ) clusters, and convolve with 
    freq.-dependent beams.
    """
    nFreq = len(beams)
    nCluster = cluster_set.Ncl
    maps = []
    
    # TSZ maps
    _tmaps = [cluster_set.get_cluster_map(i, maptype='tsz')[0]
                for i in range(nCluster)]
    tmaps = [[ applyBeam(_tmaps[i], beams[f]) for f in range(nFreq) ] 
                                              for i in range(nCluster) ]
    maps.append(tmaps)
    
    # KSZ maps
    if use_ksz:
        _kmaps = [cluster_set.get_cluster_map(i, maptype='ksz')[0] 
                    for i in range(nCluster)]
        kmaps = [[ applyBeam(_kmaps[i], beams[f]) for f in range(nFreq) ] 
                                                    for i in range(nCluster) ]
        maps.append(kmaps)
    return maps
    
def build_vcov_inverse(vcov):
    """
    Invert velocity covariance matrix.
    """
    ivcov = numpy.linalg.inv(vcov)
    return ivcov


def computeX0(linsys_setup):
    """
    Compute initial guess for solution (CMB = 0 and cluster amp. = 1).
    """
    datamaps, ninvs, beams, freqs, power_2d, precond_2d, clumaps, g_nu, \
                                           map_prop, vcov, ivcov = linsys_setup
    x = 0*datamaps[0].data
    
    # Monopole amplitude
    x = numpy.append(x, 0)
    
    # TSZ amplitudes
    for ic in range(len(clumaps[0])):
      x = numpy.append(x, 1.)
    
    # KSZ amplitudes
    if len(clumaps) == 2:
      for ic in range(len(clumaps[1])):
        x = numpy.append(x, 0.)
    return x


def computeB(linsys_setup):
    """
    Compute RHS of linear system to solve (see Eq. 22 of Eriksen et al. 2008).
    This includes projected data, white noise realisation, etc.
    """
    datamaps, ninvs, beams, freqs, power_2d, precond_2d, clumaps, g_nu, \
                                           map_prop, vcov, ivcov = linsys_setup
    nx, ny, pixScaleX, pixScaleY = map_prop
    nFreq = len(g_nu); nCluster = len(clumaps[0])
    ksz = False
    if len(clumaps)==2: ksz = True
    
    def computeCMBY(d0):
        """
        For CMB, y = S^1/2 A N^-1 d, where S is CMB signal covariance matrix (Cl's)
        """
        # N.B. Reshaping operations required to go between 2D pixel arrays and 
        # 1D vector (for linear system)
        d2 = 0
        for freq in range(nFreq):
            d1 = d0[freq].data.copy().reshape((ny,nx))
            d1 *= ninvs[freq]
            a_l = fft.fft(d1,axes=[-2,-1])
            a_l *= beams[freq]*precond_2d
            d1 = numpy.real(fft.ifft(a_l,axes=[-2,-1],normalize=True))
            d1 = numpy.reshape(d1,(nx*ny))
            d2 += d1
        return d2
    
    def computeClusterY(d0):
        """
        For cluster, y = F^T A^T N^-1 d, where F is TSZ spatial template for cluster.
        """
        d2 = numpy.zeros(nCluster)
        for ic in range(nCluster):
            for freq in range(nFreq):
                d1 = d0[freq].data.copy().reshape((ny, nx))
                d2[ic] += numpy.sum(d1 * ninvs[freq] * clumaps[0][ic][freq] * g_nu[freq])
        return d2
    
    def computeClusterKSZY(d0):
        """
        For cluster, y = K^T A^T N^-1 d, where K is KSZ spatial template for cluster.
        """
        d2 = numpy.zeros(nCluster)
        for ic in range(nCluster):
            for freq in range(nFreq):
                d1 = d0[freq].data.copy().reshape((ny, nx))
                d2[ic] += numpy.sum(d1 * ninvs[freq] * clumaps[1][ic][freq])
        return d2
    
    def computeMonopoleY(d0):
        """
        Overall monopole amplitude.
        """
        d2 = 0
        for freq in range(nFreq):
            d1 = d0[freq].data.copy().reshape((ny, nx))
            d2 += numpy.sum(d1 * ninvs[freq])
        return(d2)
    
    
    # CMB realisation; convolve white noise map with beam and multiply by 
    # signal covmat S^1/2 in harmonic space
    b0 = numpy.random.randn(ny,nx)
    a_l = numpy.fft.fft2(b0, b0.shape)
    a_l *= precond_2d * power_2d**(-0.5)
    b0 = numpy.fft.irfft2(a_l, b0.shape)
    
    # Calculate per-band noise realisation.
    # Multiply by pixel-space N^1/2, convolve with beam, and sum over 
    # cluster pixels to get RHS
    b1 = 0; b4 = 0
    b2 = numpy.zeros(nCluster)
    if ksz: b3 = numpy.zeros(nCluster)
    
    for freq in range(nFreq):
        _b = numpy.random.randn(ny,nx) * ninvs[freq]**0.5
        a_l = numpy.fft.fft2(_b) * beams[freq] * precond_2d
        b1 += numpy.fft.irfft2(a_l, _b.shape)
        b4 += numpy.sum(_b)
        for ic in range(nCluster):
            b2[ic] += numpy.sum( _b * g_nu[freq] * clumaps[0][ic][freq] )
            if ksz: b3[ic] += numpy.sum( _b * clumaps[1][ic][freq] )

    b0 = numpy.reshape(b0,(nx*ny))
    b1 = numpy.reshape(b1,(nx*ny))
    
    # Add prior term for KSZ
    if ksz:
        sqrt_ivcov = numpy.linalg.cholesky(ivcov)
        omega_1 = numpy.random.normal(size=nCluster)
        b3 += numpy.dot(sqrt_ivcov, omega_1)
    
    # Compute CMB and cluster data parts of b
    b_CMB = computeCMBY(datamaps) + b0 + b1
    b_mono = computeMonopoleY(datamaps) + b4
    b_tsz = computeClusterY(datamaps) + b2
    if ksz: b_ksz = computeClusterKSZY(datamaps) + b3
    
    # Return total b vector (Ncmbpix + 1 + (1|2)*Ncluster elements in vector)
    b = numpy.append(b_CMB, b_mono)
    b = numpy.append(b, b_tsz)
    if ksz: b = numpy.append(b, b_ksz)
    return b


def preCondConjugateGradientSolver(b, x, linsys_setup, eps, i_max, plotInterval, mapDir):
    """
    Solve linear system of equations with preconditioned CG solver.
    """
    datamaps, ninvs, beams, freqs, power_2d, precond_2d, clumaps, g_nu, \
                                           map_prop, vcov, ivcov = linsys_setup
    nx, ny, pixScaleX, pixScaleY = map_prop
    nCluster = len(clumaps[0])
    ksz = False
    if len(clumaps)==2: ksz=True
    
    
    # Calculate residual r = b - (A^-1) x
    r = b - applyMat(x, linsys_setup)
    d = r


    delta_new = numpy.inner(r,r)
    



    delta_o = delta_new
    delta_array = numpy.zeros(shape=(i_max))
    
    # Iterate CG solver until converged
    i = 0
    #i_max = 300
    while (i < i_max) and (delta_new > delta_o*eps**2.):
        if i==0: t = time.time()
        
        if i%plotInterval == 0 and i != 0:
            print "\tNumber of iterations in the CG:", i
            x0 = x[:nx*ny] # CMB
            x1 = x[nx*ny:nx*ny+1] # Monopole
            x2 = x[nx*ny+1:nx*ny+1+nCluster] # TSZ
            if ksz: x3 = x[nx*ny+1+nCluster:nx*ny+1+2*nCluster]
            print "\tMonopole:", x1
            print "\tTSZ:", x2
            if ksz: print "\tKSZ:", x3
            
            x0.shape = (ny,nx)
            a_l = numpy.fft.fft2(x0)
            a_l *= precond_2d
            x_test = numpy.real(numpy.fft.ifft2(a_l))
            plot(x_test,mapDir+'/CMB_%d.png'%i,'Reconstructed CMB', range=(-250., 250.))
            print delta_new, delta_o*eps**2.

        q = applyMat(d, linsys_setup)
        alpha = delta_new / (numpy.inner(d,q))
        x += alpha * d

        # What does this do? It's always false.
        if i/50. < numpy.int(i/50):
            r = b - applyMat(x, linsys_setup)
        else:
            r = r - alpha*q
        
        delta_old = delta_new
        delta_new = numpy.inner(r,r)
        beta = delta_new/delta_old
        d = r + beta * d
        #if i==0: print "\tEach iteration takes:", time.time()-t
        i += 1

    x0 = x[:nx*ny].reshape((ny, nx))
    x1 = x[nx*ny:nx*ny+1]
    x2 = x[nx*ny+1:nx*ny+1+nCluster]
    if ksz:
        x3 = x[nx*ny+1+nCluster:nx*ny+1+2*nCluster]
    else:
        x3 = None
    
    a_l = numpy.fft.fft2(x0) * precond_2d
    x0 = numpy.real(numpy.fft.ifft2(a_l))
    
    # CMB, monopole, TSZ, KSZ
    return x0, x1, x2, x3


def applyMat(my_map, linsys_setup):
    """
    Multiply my_map by the matrix operator, defined in Eq. 19 of Eriksen 
    et al. (2008).
    """
    
    datamaps, ninvs, beams, freqs, power_2d, precond_2d, clumaps, g_nu, \
                                           map_prop, vcov, ivcov = linsys_setup
    
   

    nx, ny, pixScaleX, pixScaleY = map_prop
    nFreq = len(g_nu); nCluster = len(clumaps[0])

    #Always apply beam * precond
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
    
    """
    def apply_tsz_tsz(d0): # DONE
        \"""
        Apply (F^T A^T N^-1 A F) x
        \"""
        d1 = numpy.zeros(nCluster)
        mat = numpy.zeros((nCluster, nCluster))
        # TODO: This could probably be more efficient (e.g. using np.outer)
        for freq in range(nFreq):
          for ic in range(nCluster):
            for jc in range(0, ic+1):
              mat[ic,jc] = numpy.sum(  ninvs[freq] * g_nu[freq]**2. \
                                     * clumaps[0][ic][freq] * clumaps[0][jc][freq] )
              if ic != jc: mat[jc,ic] = mat[ic,jc]
          d1 += numpy.dot(mat, d0)
        return d1
    
    def apply_ksz_ksz(d0): # DONE
        \"""
        Apply (K^T A^T N^-1 A K) x
        \"""
        # FIXME: Missing factor of ivcov
        d1 = numpy.zeros(nCluster)
        mat = numpy.zeros((nCluster, nCluster))
        # TODO: This could probably be more efficient (e.g. using np.outer)
        for freq in range(nFreq):
          for ic in range(nCluster):
            for jc in range(0, ic+1):
              mat[ic,jc] = numpy.sum(  ninvs[freq]  \
                                     * clumaps[1][ic][freq] * clumaps[1][jc][freq] )
              if ic != jc: mat[jc,ic] = mat[ic,jc]
          d1 += numpy.dot(mat, d0)
        d1 += numpy.dot(ivcov, d0) # Add prior term
        return d1
    """
    
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
        
          b_lt += fft.rfft(mct,axes=[-2,-1]) * beam_prec[f]
          b_lm += fft.rfft(dm * ninvs[f],axes=[-2,-1])  * beam_prec[f]
          if ksz: b_lk += fft.rfft(mck,axes=[-2,-1]) * beam_prec[f]
        mct = fft.irfft(b_lt,axes=[-2,-1],normalize=True).reshape((nx*ny,))
        mcm = fft.irfft(b_lm,axes=[-2,-1],normalize=True).reshape((nx*ny,))
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
              dmk += numpy.sum( ninvs[f] * g_nu[f] * clumaps[1][ic][f] * k0[ic] )
              dkm[ic] += numpy.sum( ninvs[f] * g_nu[f] * clumaps[1][ic][f] * m0 )
            
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
            
            # Add prior term to KSZ-KSZ part
            dkk += numpy.dot(ivcov, k0)
            
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


#sys.exit()
    return x_new

