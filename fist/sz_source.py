from units import *
from compact_source import *
import scipy.integrate
import scipy.interpolate

class SZCluster(Source):

  def __init__(self, pos, params, template, paramnames=None, cosmo=None):
    """
    Generic SZ cluster base class, with both TSZ and KSZ profiles.
    
    The following methods/properties should be overridden by the child class:
        self.tsz_profile()
        self.ksz_profile()
        self.tsz_amp
        self.ksz_amp
    """
    # Initialise parent class
    super(SZCluster, self).__init__(pos, params, template, 
                                    paramnames=paramnames, cosmo=cosmo)
    
    # Sanity checks to make sure SZ class can initialise properly
    if self.cosmo is None:
        raise ValueError("SZCluster(): Cosmology not specified (use 'cosmo' arg).")
    
    # Set SZ cluster parameters
    self.update_params(params, paramnames)
  
  def update_params(self, params, paramnames=None):
    """
    Update SZ cluster profile parameters.
    
    Parameters
    ----------
    params : array_like
        Array of parameter values, to update the parameters for this SZ 
        cluster. If paramnames==None, this will be interpreted as an ordered 
        list of parameters.
    
    paramnames : list_like of str, optional
        Ordered list of parameter names, used to label the 'params' array. 
        If None, 'params' is interpreted as an ordered list. Default: None.
    """
    self.z = params[pn.index('z')] if pn is not None else params[0]
    self.r500 = params[pn.index('r500')] if pn is not None else params[1]
    
    # Set required amplitude properties to trivial values
    self.tsz_amp = 1.
    self.ksz_amp = 1.
  
  def spectrum(self, nu, type='tsz'):
    """
    Return frequency scaling, g_nu, suitable for brightness temperature 
    fluctuation.
    
    Parameters
    ----------
    nu : float
        Frequency, in GHz.
    
    type : str, optional
        Which type of signal to return the frequency scaling for. Valid options 
        are 'tsz' or 'ksz'. Default: 'tsz'.
    
    Returns
    -------
    g_nu : float
        Frequency scaling.
    """
    # Sanity check on signal type
    type = type.lower()
    if type not in ('tsz', 'ksz'): raise ValueError("Invalid 'type': use 'tsz' or 'ksz'.")
    
    # Return appropriate g_nu
    if type == 'tsz':
        x = NU_SCALE * nu
        return x**2. * np.exp(x)*(x/np.tanh(x/2.) - 4.) / (np.exp(x) - 1.)**2.
    else:
        return np.ones_like(nu)
        
  
  def profile(self, type='tsz'):
    """
    Spatial profile as a fn. of distance from the center of the SZ cluster.
    
    Parameters
    ----------
    type : str, optional
        Which type of spatial template to return, if the source has more than 
        one type to choose from.
    
    Returns
    -------
    profile : function()
        Spatial profile interpolation function, with call signature 
        profile(theta), where theta is in degrees.
    """
    # Sanity check on signal type
    type = type.lower()
    if type not in ('tsz', 'ksz'): raise ValueError("Invalid 'type': use 'tsz' or 'ksz'.")
    
    # Get appropriate profile function and amplitude for this type
    if type == 'ksz':
        amp = self.ksz_amp
        prof = self.ksz_profile
    else:
        amp = self.tsz_amp
        prof = self.tsz_profile
    
    # Define r (transverse dist.) array and sample length
    l = np.logspace(-3., np.log10(50.*self.r500), 1000)
    d = l[::10] # Only sample every 10 resolution elements
    
    # Calculate integral along LOS through profile
    integ_prof = [ scipy.integrate.simps(prof(np.sqrt(l**2. + _d**2.)), l)
                   for _d in d ]
    integ_prof = 2. * np.array(integ_prof) # Factor of 2 from +/-l symmetry
    
    # Convert physical distance to angle (in degrees), and add zero-angle value
    dA = self.cosmo.r(self.z) / (1. + self.z)
    th = (d / dA) * 180. / np.pi
    th = np.concatenate(([0.,], th))
    integ_prof = np.concatenate(([integ_prof[0],], integ_prof))
    
    # Interpolate and return
    interp_prof = scipy.interpolate.interp1d(th, amp*integ_prof, kind='linear', 
                                             bounds_error=False, fill_value=0.)
    return interp_prof
    
  def tsz_profile(self, r):
    """
    TSZ radial profile, \propto P_e(r). This will be integrated along the LOS 
    by self.profile().
    """
    return np.zeros_like(r)

  def ksz_profile(self, r):
    """
    KSZ radial profile, \propto n_e(r). This will be integrated along the LOS 
    by self.profile().
    """
    return np.zeros_like(r)



class GNFWCluster(SZCluster):

  def __init__(self, pos, params, template, paramnames=None, cosmo=None):
    """
    SZ cluster source, with TSZ and KSZ profiles based on a GNFW model.
    
    Parameters
    ----------
    pos : tuple(2) of float
        Position of the SZ cluster center, (RA, Dec).
        
    params : array_like(7) of float
        Array of cluster parameter values, for the parameters: 
            (z, M500, alpha, beta, gamma, c500, P0).
        (Units: M500 ~ Msun)
        Specify 'paramnames' to use a different ordering of the array.
    
    template : Flipper LiteMap
        Template LiteMap used to define coordinate system and pixel grid.
    
    paramnames : list of str, optional
        Ordered list of parameter names for the array 'params'. If paramnames 
        is not specified, the 'params' array will be assumed to be ordered as: 
            (z, M500, alpha, beta, gamma, c500, P0)
    
    cosmo : Cosmology() object
        Convenience class to calculate cosmological functions. Required.
    """
    # Initialise parent class
    super(GNFWCluster, self).__init__(pos, params, template, 
                                    paramnames=paramnames, cosmo=cosmo)
    
  def update_params(self, params, paramnames=None):
    """
    Update SZ cluster profile parameters.
    
    Parameters
    ----------
    params : array_like
        Array of parameter values, to update the parameters for this SZ 
        cluster. If paramnames==None, this will be interpreted as an ordered 
        list of parameters, with order:
            'z', 'M500', 'alpha', 'beta', 'gamma', 'c500', 'P0'
    
    paramnames : list_like of str, optional
        Ordered list of parameter names, used to label the 'params' array. 
        If None, 'params' is interpreted as an ordered list. Default: None.
    """
    # Sanity check on parameters
    pn = paramnames
    plist = ['z', 'M500', 'alpha', 'beta', 'gamma', 'c500', 'P0']
    if pn is not None:
        for pp in plist:
            if pp not in pn: raise KeyError('%s parameter not found.' % pp)
    
    # Set parameter values
    self.z = params[pn.index('z')] if pn is not None else params[0]
    self.M500 = params[pn.index('M500')] if pn is not None else params[1]
    self.alpha = params[pn.index('alpha')] if pn is not None else params[2]
    self.beta = params[pn.index('beta')] if pn is not None else params[3]
    self.gamma = params[pn.index('gamma')] if pn is not None else params[4]
    self.c500 = params[pn.index('c500')] if pn is not None else params[5]
    self.P0 = params[pn.index('P0')] if pn is not None else params[6]
    
    # Update derived parameters
    self.update_derived()
  
  
  def update_parameter(self, paramname, value):
    """
    Update the value of a specific parameter.
    """
    pname = paramname.lower()
    
    # Update the chosen parameter
    if pname == 'z':
        self.z = value
    elif pname == 'm500':
        self.M500 = value
    elif pname == 'alpha':
        self.alpha = value
    elif pname == 'beta':
        self.beta = value
    elif pname == 'gamma':
        self.gamma = value
    elif pname == 'c500':
        self.c500 = value
    elif pname == 'p0':
        self.P0 = value
    else:
        raise KeyError("Parameter '%s' is not recognised, or is not an input parameter." \
                       % paramname)
    
    # Update derived parameters
    self.update_derived()
  
  
  def update_derived(self):
    """
    Update derived parameters from current set of input parameters.
    """
    h = self.cosmo.cosmo['h']
    hfac = h**(2./3.) * (self.cosmo.H(self.z) / (100.*h))**(8./3.)
    
    # Derived using M_500 = 4pi/3 (r_500)^3 * 500 * rho_crit(z). Units: Mpc.
    self.r500 = ( (self.M500 / 5.81e10) / (self.cosmo.H(self.z))**2. )**(1./3.)
    
    # Derive in small-angle approximation, r_500 / D_A. Units: arcmin
    self.th500 = self.r500 / (self.cosmo.r(self.z) / (1. + self.z)) \
               * (180. * 60. / np.pi)
    
    # Derived using Eq. 2 of arXiv:astro-ph/0703661. Units: kg/m/s^2
    self.P500 = 1.45e-22 * (self.M500)**(2./3.) * hfac
    
    # Overall mass scaling of NFW profile. Derived by imposing M(<r_500) = M_500
    # Units: M_sun.
    self.Mstar = self.M500 / (np.log(1. + self.c500) - self.c500/(1. + self.c500))
    
    # Overall TSZ and KSZ profile amplitudes
    # Units: tsz_amp ~ Mpc^-1; ksz_amp ~ Mpc^-2 / (km/s)
    self.tsz_amp = 3.63e-15 * self.P0 * self.M500**(2./3.) * hfac
    self.ksz_amp = -0.253 * self.M500**(-1./3.) * hfac \
                   * (np.log(1. + self.c500) - self.c500/(1. + self.c500))

  def tsz_profile(self, r):
    """
    TSZ radial profile, \propto P_e(r). This will be integrated along the LOS 
    by self.profile(). Units: Dimensionless.
    """
    return self.fp(r)

  def ksz_profile(self, r):
    """
    KSZ radial profile, \propto n_e(r). This will be integrated along the LOS 
    by self.profile(). Units: Mpc.
    """
    return r * self.fp(r) * self.fn(r) / self.fm(r)

  def fp(self, r):
    """
    Dimensionless profile for electron pressure, f_P(r), where 
    P_e(r) = P_500 . P_0 . f_P(r). Expects r ~ Mpc.
    """
    x = r / (self.r500 / self.c500)
    fpinv = x**self.gamma \
          * (1. + x**self.alpha)**((self.beta - self.gamma) / self.alpha)
    return 1. / fpinv

  def fm(self, r):
    """
    Dimensionless profile for mass within a given radius, f_M(r), where 
    M(<r) = M_* . f_M(r). Expects r ~ Mpc.
    """
    x = r / (self.r500 / self.c500)
    return ( (1. + x)*np.log(1. + x) - x ) / (1. + x)

  def fn(self, r):
    """
    Dimensionless profile for log derivative of pressure profile, dlogP/dr.
    Expects r ~ Mpc.
    """
    x = r / (self.r500 / self.c500)
    return (self.gamma + self.beta * x**self.alpha) / (1. + x**self.alpha)
