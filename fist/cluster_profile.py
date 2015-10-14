#!/usr/bin/python
"""Arnaud et al. cluster pressure profile (arXiv:0910.1234)."""

import numpy as np
import scipy.integrate
import scipy.interpolate
import astLib.astCoords
from units import *

class ArnaudProfile(object):
    
    def __init__(self, params=None, precalc=False, coords=None,
                       cosmo=DEFAULT_COSMO):
        """
        Arnaud et al. cluster pressure profile (arXiv:0910.1234). Parameters 
        are: [alpha, beta, gamma, c500, P0, M500, r500, z].
        """
        
        # Set default cosmological parameters
        self.set_cosmology(cosmo)
        
        # Use "illustrative" cluster parameters if no actual params specified
        if params == None:
          print "\tNo cluster params. specified; using 'illustrative' cluster params."
          self.profile_example_cluster()
        elif len(params) == 3:
          self.profile_arnaud_bestfit()
          self.set_physical_params(params)
        else:
          self.set_profile_params(params)
          
        # Set cluster coordinates in (l, b) and (RA, Dec)
        if coords == None:
          self.update_coords(0., 0.)
        else:
          self.update_coords(coords[0], coords[1])
        
        # Set mass scaling, alpha_p, defined in Arnaud Eq. 7.
        # alpha_p = 1/alpha_MYx - 5/3, alpha_MYx = 0.561 (X-ray best-fit)
        self.alpha_p =  0.12
        
        # Set SZ profile integration/interpolation limits (units of R500)
        self.bmax = 24. # Interpolation limit
        self.bmaxc = 25. # Integration limit
        
        # Pre-calculate some values
        #self._P500 = self.P500(self.z)
        self._P500 = 0. # FIXME
        
        #self.profile_arnaud_bestfit()
        
    
    ############################################################################
    # Parameter handling
    ############################################################################
    
    def profile_arnaud_bestfit(self):
        """Use best-fit universal profile parameters from Arnaud Eq. 12."""
        # Universal profile parameters
        self.P0 = 8.403 * (self.h70)**(-1.5)
        self.c500 = 1.177
        self.gamma = 0.3081
        self.alpha = 1.0510
        self.beta = 5.4905
    
    def profile_example_cluster(self):
        """
        Pick illustrative values of all required parameters, for a "typical" 
        cluster. M500 = 1e14 Msun; R500 = 1.0 Mpc; z = 0.5.
        """
        self.profile_arnaud_bestfit() # Set default shape parameters
        self.M500 = 1e14
        self.r500 = 1.
        self.z = 0.5
    
    def set_profile_params(self, params):
        """
        Set all parameters of the cluster, including shape and basic physical 
        properties (mass, redshift).
        Expects params = [alpha, beta, gamma, c500, P0, M500, R500, z].
        """
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.c500 = params[3]
        self.P0 = params[4]
        self.M500 = params[5]
        self.r500 = params[6]
        self.z = params[7]
    
    def set_shape_params(self, params):
        """
        Set only the shape parameters (i.e. leave physical parameters fixed at 
        their current values). Expects params = [alpha, beta, gamma, c500, P0].
        """
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.c500 = params[3]
        self.P0 = params[4]
    
    def set_physical_params(self, params):
        """
        Set only the physical parameters. Expects params = [M500, r500, z].
        """
        self.M500 = params[0]
        self.r500 = params[1]
        self.z = params[2]
    
    def get_profile_params(self):
        """
        Return a vector of profile parameters in the standard order:
        Returns: list(alpha, beta, gamma, c500, P0, M500, R500, z)
        """
        params = [self.alpha, self.beta, self.gamma, self.c500, self.P0, 
                  self.M500, self.r500, self.z]
        return params
    
    def update_coords(self, l, b):
        """
        Update galactic (l,b) and RA/dec coordinates of cluster.
        """
        self.l = l
        self.b = b
        self.ra, self.dec = astLib.astCoords.convertCoords(
                              "GALACTIC", "J2000", self.l, self.b, epoch=2000.)
        
    
    ############################################################################
    # Pressure profiles
    ############################################################################
    
    # FIXME: Can be optimised (slow at the moment)
    
    def alpha_pp(self, x):
        """Running of scaling with mass, from Arnaud Eq. 9."""
        y = (2.*x)**3.
        return 0.10 - ( (self.alpha_p + 0.10) * y / (1. + y) )
    
    def p(self, x):
        """Dimensionless cluster pressure profile (GNFW), from Arnaud Eq. 11."""
        y = self.c500*x
        pp = y**self.gamma * (1. + y**self.alpha)**((self.beta - self.gamma)/self.alpha)
        return self.P0 / pp
    
    def P500(self, z):
        """Pressure at radius where delta=500, from Arnaud Eq. 5."""
        # FIXME: Factor of h70^2 in Eq. 5 needs to be removed to match their 
        # results in Table C.1. Why is this?
        # Also note that, according to their Figs. 5--8, P(R500) \neq P500!
        pp = 1.65e-3 * (self.h(z))**(8./3.) #* self.h70**2.
        return pp * (self.M500 * self.h70 / 3e14)**(2./3.)
    
    def P(self, r):
        """
        Extended cluster pressure profile, assuming standard cosmological 
        evolution, from Arnaud Eq. 13. Units: keV cm^-3.
        """
        x = r/self.r500
        p500 = self.P500(self.z)
        aa = self.alpha_p + self.alpha_pp(x)
        return p500 * self.p(x) * (self.M500 * self.h70 / 3e14)**aa
    
    def xP(self, r):
        """
        Extended cluster pressure profile, assuming standard cosmological 
        evolution, from Arnaud Eq. 13. Dimensionless; multiply by P500(z).
        """
        x = r/self.r500
        aa = self.alpha_p + self.alpha_pp(x)
        return self.p(x) * (self.M500 * self.h70 / 3e14)**aa
    
    ############################################################################
    # Cosmology
    ############################################################################
    
    def set_cosmology(self, cosmo):
        """
        Set the cosmological parameters to use when evolving the model with 
        redshift.
        """
        self.cosmo = cosmo
        self.h70 = cosmo['h'] # Hubble parameter, H0 = 100h km/s/Mpc
        self.Om = cosmo['omega_M_0'] # Omega_matter
        self.Ol = cosmo['omega_lambda_0'] # Omega_Lambda
    
    def h(self, z):
        """Relative Hubble rate as a function of redshift."""
        # See definition at end of Section 1, p2 of Arnaud et al.
        return np.sqrt(self.Om*(1.+z)**3. + self.Ol)
    
    
    ############################################################################
    # SZ observables
    ############################################################################
    
    def Ysph(self, R):
        """
        Integrate pressure profile over cluster volume, to give total SZ 
        Y-parameter, Arnaud Eq. 14.
        """
        pp, err = scipy.integrate.quad(lambda x: self.xP(x)*x**2., 0., R)
        y = 4. * np.pi * Y_SCALE * pp * self._P500
        return y

    def Ycyl(self, R):
        """
        Integrate pressure profile along a pencil beam through the cluster to 
        give SZ Y-parameter, Arnaud Eq. 15.
        """
        # FIXME: Not implemented yet
        raise NotImplementedError("ArnaudProfile.Ycyl not implemented.")
        return 0.
    
    def dI_SZ(self, l):
        """
        SZ surface brightness profile perturbation, from Rephaeli Eqs. 6, 8.
        """
        # Valid in RJ limit, x << 1
        
        # Get dimensionless frequency
        x = X_SCALE * NU_WMAP_W / ( T_CMB * (1. + self.z) )
        
        # Integrate pressure profile at fixed radius
        xmax = 5. * self.r500; xmin = -xmax
        pp, err = scipy.integrate.quad( lambda x: self.xP(np.sqrt(x**2. + l**2.)), 
                                        xmin, xmax )
        y = Y_SCALE * pp * self._P500
        dI = -2. * y * (1. + 0.5*x)
        return dI
    
    
    ############################################################################
    # SZ profiles
    ############################################################################
    
    def Tloken(self, x):
        """Loken universal temperature profile, in keV."""
        return 11.2 * (self.r500*0.7)**2. * ( 1. + 0.75*x)**(-1.6)
    
    def _ig_tsz(self, x, b):
        """Integrand for tSZ profile."""
        return self.P(x*self.r500) * (x / np.sqrt(x**2. - b**2.))
    
    def _ig_ksz(self, x, b):
        """Integrand for kSZ profile."""
        return self.P(x*self.r500) * (x / np.sqrt(x**2. - b**2.)) / self.Tloken(x)

    def tsz_spectrum(self, nu):
        """Spectral dependence of TSZ effect."""
        x = NU_SCALE * nu # Frequency/temperature
        #g_nu = ( x*(np.exp(x) + 1.) / (np.exp(x) - 1.) ) - 4. # tSZ spectral dependence
        g_nu = x**2. * np.exp(x) * (x/np.tanh(x/2.) - 4.) / (np.exp(x) - 1.)**2.
        return g_nu

    def tsz_profile(self, nu=None):
        """
        Return interpolation fn. for tSZ profile as a function of r [Mpc].
        """
        bb = np.linspace(0.0, self.bmax, 150) # Range of impact parameters
        rr = bb * self.r500
        
        # Interpolate the radial pressure profile, P
        N_X_SAMP = 1200 # Increase this for more accurate integration
        _r = np.logspace(-4, np.log10(self.bmax*self.r500), 200)
        #_r = np.linspace(1e-4, self.bmax*self.r500, 250)
        _P = self.P(_r)
        Pinterp = scipy.interpolate.interp1d(_r, _P, kind='linear', 
                                             bounds_error=False, fill_value=0.)
        
        # Sample the integrand and do Simpson-rule integration over samples
        ig_tsz = lambda x, b: Pinterp(x*self.r500) * (x / np.sqrt(x**2. - b**2.))
        _x = [ np.logspace(np.log10(b+1e-4), np.log10(self.bmaxc), N_X_SAMP) 
                for b in bb ]
        ysz = [ scipy.integrate.simps(ig_tsz(_x[i], bb[i]), _x[i]) 
                  for i in range(bb.size) ]
        
        # Spectral dependence and Y_SZ pre-factors
        if nu == None:
          g_nu = 1.
        else:
          g_nu = self.tsz_spectrum(nu)
        fac_ysz = (2. * 2. * 2.051 / 511.) * self.r500
        ysz = g_nu * fac_ysz * np.array(ysz)
        
        # Interpolate and return
        interp = scipy.interpolate.interp1d( rr, ysz, kind='linear', 
                                             bounds_error=False, fill_value=0.0 )
        return interp
    
    def ksz_profile(self, nu=None):
        """
        Return interpolation fn. for KSZ profile as a function of r. Units of 
        uK/(km/s), so need to multiply by velocity in km/s to give physical map 
        on the sky.
        """
        # TODO: Use faster interpolation
        bb = np.linspace(0.0, self.bmax, 100) # Range of impact parameters
        rr = bb * self.r500
    
        fac_ksz = 2. * 2.051 * self.r500 / 3e5
        ksz = map(lambda b: scipy.integrate.quad(self._ig_ksz, b+0.0001, 
                                            self.bmaxc, args=(b,))[0], bb)
        ksz = fac_ksz * np.array(ksz)
        interp = scipy.interpolate.interp1d(rr, ksz, bounds_error=False, fill_value=0.0)
        return interp

    def tsz_profile_highacc(self, nu=None):
        """
        Return interpolation fn. for TSZ profile as a function of r [Mpc]. This 
        uses a higher accuracy numerical integration than tsz_profile(), which 
        just uses the Simpson rule.
        """
        bb = np.linspace(0.0, self.bmax, 100) # Range of impact parameters
        rr = bb * self.r500
        ysz = map(lambda b: scipy.integrate.quad(self._ig_tsz, b+0.0001,
                                            self.bmaxc, args=(b,))[0], bb)
        if nu == None:
          g_nu = 1. # Factor-out spectral dependence
        else:
          g_nu = self.tsz_spectrum(nu)
        fac_ysz = (2. * 2. * 2.051 / 511.) * self.r500
        ysz = g_nu * fac_ysz * np.array(ysz)
        interp = scipy.interpolate.interp1d( rr, ysz, kind='linear', 
                                             bounds_error=False, fill_value=0.0 )
        return interp

    def ksz_profile_highacc(self, v, nu=1.):
        """Return interpolation fn. for KSZ profile as a function of r."""
        bb = np.linspace(0.0, self.bmax, 100) # Range of impact parameters
        rr = bb * self.r500
    
        fac_ksz = 2. * 2.051 * v * self.r500 / 3e5
        ksz = map(lambda b: scipy.integrate.quad(self._ig_ksz, b+0.0001, 
                                            self.bmaxc, args=(b,))[0], bb)
        ksz = fac_ksz * np.array(ksz)
        interp = scipy.interpolate.interp1d(rr, ksz, bounds_error=False, fill_value=0.0)
        return interp
        
    def tsz_profile_for_params(self, params, nu=None):
        """
        Return interpolation fn. for tSZ profile as a function of r [Mpc], 
        given a set of shape+physical parameters.
        """
        # Save current parameters
        params0 = self.get_profile_params()
        
        # Temporarily set new profile parameters and calculate new profile
        self.set_profile_params(params)
        prof = self.tsz_profile(nu)
        
        # Reset to original params and return result
        self.set_profile_params(params0)
        return prof
    
    def ksz_profile_for_params(self, params):
        """
        Return interpolation fn. for kSZ profile as a function of r [Mpc], 
        given a set of shape+physical parameters.
        """
        # Save current parameters
        params0 = self.get_profile_params()
        
        # Temporarily set new profile parameters and calculate new profile
        self.set_profile_params(params)
        prof = self.ksz_profile()
        
        # Reset to original params and return result
        self.set_profile_params(params0)
        return prof
        
