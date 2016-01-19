#!/usr/bin/python
"""
Cluster shape- and position-sampling MCMC.
"""

import numpy as np
from .map_tools import *
from .utils import *
from mpi4py import MPI as mpi


def residual_maps_for_cluster( j, tsz_profiles, ksz_profiles, cmbmap, tsz_amps, 
                               ksz_amps, cl_set, expt_setup ):
    """
    Get residual map for cluster j, which is the data map with CMB and other 
    clusters subtracted.
    """
    datamaps, ninvs, beams, freqs = expt_setup
    
    # Add all clusters (apart from this one) to blank map
    clmap = 0
    if ksz_profiles[0] is not None:
        ksz_clmap = 0
    
    for i in range(cl_set.Ncl):
        if i != j:
            ra = cl_set.clusters[i].ra
            dec = cl_set.clusters[i].dec
            z = cl_set.clusters[i].z
            clmap += tsz_amps[i] * cl_set.get_cluster_map_for_profile(
                                                   tsz_profiles[i], ra, dec, z )
            if ksz_profiles[0] is not None:
                ksz_clmap += ksz_amps[i] * cl_set.get_cluster_map_for_profile(
                                                   ksz_profiles[i], ra, dec, z )
    
    # Form frequency-dependent residual maps
    resmaps = []
    for k in range(len(freqs)):
        m = cl_set.tsz_spectrum(nu=freqs[k]) * clmap
        if ksz_profiles[0] is not None:
            m += ksz_clmap
        
        # Add CMB to map, convolve with beam, and then subtract data to leave residual
        m += cmbmap
        m = applyBeam(m, beams[k])
        m = datamaps[k].data - m
        resmaps.append(m)
    return resmaps


def skymap_for_cluster( i, tsz_profile, ksz_profile, cl_set, 
                        ra=None, dec=None, z=None, beam_tmpl=None ):
    """
    Construct a map for a cluster at a given position on the sky, using a given 
    cluster radial profile, with (optionally) a beam convolution applied.
    """
    # Get position of cluster
    if ra==None:  ra = cl_set.clusters[i].ra
    if dec==None: dec = cl_set.clusters[i].dec
    if z==None:   z = cl_set.clusters[i].z
    
    # Get map for cluster and convolve with beam
    tsz_clmap = cl_set.get_cluster_map_for_profile(tsz_profile, ra, dec, z)
    if beam_tmpl is not None:
        tsz_clmap = applyBeam(tsz_clmap, beam_tmpl)
    
    if ksz_profile is not None:
        ksz_clmap = cl_set.get_cluster_map_for_profile(ksz_profile, ra, dec, z)
        if beam_tmpl is not None:
            ksz_clmap = applyBeam(ksz_clmap, beam_tmpl)
        return tsz_clmap, ksz_clmap
    else:
        return tsz_clmap, None


def get_cluster_profiles(params, cl_set, ksz=False, comm=None):
    """
    Pre-calculate radial profiles for all clusters, given a list of parameters 
    for each cluster.
    """
    # NOTE: This is relatively computationally intensive (~0.5sec/cluster on lynx).
    # FIXME: Need to optimise pressure profile fn. in cluster_profile
    # TODO: Allow this routine to accept just one set of parameters and apply 
    # it to all clusters.
    # TODO: Parallelise this
    
    """
    # FIXME: Can't deal with function objects, like interpolating functions
    # Setup MPI
    rank = None
    if comm is not None: rank = comm.Get_rank()
    my_cluster_idxs = idxs_for_worker(cl_set.Ncl, rank, comm)
    
    # Calculate profiles in parallel and distribute results to all processes
    my_profiles = []
    for i in my_cluster_idxs:
        prof = cl_set.clusters[i].tsz_profile_for_params(params[i])
        my_profiles.append(prof)
    profiles = gather_list_from_all(my_profiles, cl_set.Ncl, comm)
    """
    tsz_profiles = [ cl_set.clusters[i].tsz_profile_for_params(params[i]) 
                     for i in range(cl_set.Ncl) ]
    if ksz:
        ksz_profiles = [ cl_set.clusters[i].ksz_profile_for_params(params[i]) 
                         for i in range(cl_set.Ncl) ]
        return tsz_profiles, ksz_profiles
    else:
        return tsz_profiles, [None for i in range(cl_set.Ncl)]


def set_cluster_profile_params(params, cl_set):
    """
    Set profile parameters for all clusters.
    """
    for i in range(cl_set.Ncl):
        cl_set.clusters[i].set_profile_params(params[i])

def set_cluster_positions(positions, cl_set):
    """
    Set positions of all clusters.
    """
    for i in range(cl_set.Ncl):
        l, b = positions[i]
        cl_set.clusters[i].update_coords(l, b)
    

def mh_sample_cluster_shapes( params, cov_prop, cmbmap, tsz_amps, ksz_amps, 
                              cl_set, expt_setup, max_ntries=100, comm=None ):
    """
    Sample all cluster shapes simultaneously, by making the approximation that 
    the likelihoods for each cluster are independent of one another (i.e. no 
    overlap).
    
    Parameters
    ----------
    
    params : list of array_like
        List of profile parameters for each cluster. Each set of parameters
        should be in the order [alpha, beta, gamma, c500, P0, M500, R500, z].
    
    cov_prop : array_like
        Covariance matrix for MCMC proposal distribution.
    
    cmbmap : 2D litemap
        Map of the current CMB signal sample
    
    tsz_amps, ksz_amps : array_like
        Current sample of SZ cluster amplitudes. ksz_amps is optional.
    
    cl_set : ClusterSet object
        Collection of clusters.
    
    expt_setup : tuple of lists
        Tuple containing lists of experimental data/settings for each band 
        (data map, inverse noise map, beam template, and band frequency).
        
        The structure of expt_setup is: ([datamap], [Ninv], [beam], [freq1]),
        where each list has length Nbands.
    
    max_ntries : int, optional
        Max. number of rejections to allow for each cluster before giving up 
        and quitting the sampler.
    
    comm : MPI communicator (optional)
        MPI communicator. If not defined, no parallelisation will be used.
    
    Returns
    -------
    new_params: list of array_like
        List of sampled profile parameters for each cluster.
    """
    datamaps, ninvs, beams, freqs = expt_setup
    g_nu = [cl_set.tsz_spectrum(nu=f) for f in freqs]
    
    # Setup MPI
    rank = None
    if comm is not None: rank = comm.Get_rank()
    my_cluster_idxs = idxs_for_worker(cl_set.Ncl, rank, comm)
    
    # Get profiles for current parameters, for all clusters
    use_ksz = False
    if ksz_amps is not None: use_ksz = True
    tsz_profs, ksz_profs = get_cluster_profiles(params, cl_set, ksz=use_ksz, comm=comm)
    
    # Get residual maps for each cluster
    new_params = []
    for i in my_cluster_idxs:
        #print ">>> Worker", rank, "processing cluster", i
        
        # Get residual maps and chi-squared
        r = residual_maps_for_cluster( i, tsz_profs, ksz_profs, cmbmap, tsz_amps, 
                                       ksz_amps, cl_set, expt_setup )
        tsz_clmap, ksz_clmap = skymap_for_cluster(i, tsz_profs[i], ksz_profs[i], cl_set)
        chisq0 = 0.
        for k in range(len(freqs)):
            _clmap = tsz_amps[i] * g_nu[k] * tsz_clmap
            if use_ksz:
                _clmap += ksz_amps[i] * ksz_clmap
            _clmap = applyBeam(_clmap, beams[k])
            r0 = r[k] - _clmap
            chisq0 += np.sum(r0 * ninvs[k] * r0)
        
        # Propose new values until sample is accepted
        accepted = False; ntries = 0
        while not accepted:
            
            # Propose new set of parameters and calculate residual, chisq.
            prop = np.random.multivariate_normal(params[i], cov_prop)
            prop_prof_tsz = cl_set.clusters[i].tsz_profile_for_params(prop)
            prop_prof_ksz = None
            if use_ksz:
                prop_prof_ksz = cl_set.clusters[i].ksz_profile_for_params(prop)
            tclmap_prop, kclmap_prop = skymap_for_cluster(i, prop_prof_tsz, 
                                                          prop_prof_ksz, cl_set)
            chisq1 = 0.
            for k in range(len(freqs)):
                _clmap_prop = tsz_amps[i] * g_nu[k] * tclmap_prop
                if use_ksz:
                    _clmap_prop += ksz_amps[i] * kclmap_prop
                _clmap_prop = applyBeam(_clmap_prop, beams[k])
                rp = r[k] - _clmap_prop
                chisq1 += np.sum(rp * ninvs[k] * rp)
            
            #print "\tchisq:", round(chisq1,0), round(chisq0,0), round(chisq1 - chisq0, 4)
            
            # Do Hastings accept/reject test
            u = np.random.uniform()
            mh = np.exp(-0.5*(chisq1 - chisq0))
            if mh >= u:
                print "\t", rank, "ACCEPT: %3.5f" % mh
                accepted = True
            else:
                print "\t", rank, "REJECT: %3.5f" % mh
            
            # Test to prevent infinite loop if sample can't be returned
            ntries += 1
            if ntries > max_ntries:
                print "ERROR: Failed to accept sample for cluster", i, 
                print "after", ntries, "tries. Quitting."
                comm.Abort(); raise ValueError()
                
        # Save new values of parameters
        new_params.append(prop)
    
    # Gather updated parameters from all processes
    new_params = gather_list_from_all(new_params, cl_set.Ncl, comm)
    
    # Update profile parameters and return new set of parameters
    set_cluster_profile_params(new_params, cl_set)
    return new_params


def mh_sample_cluster_positions(pos, cov_prop, cmbmap, tsz_amps, ksz_amps,
                                cl_set, expt_setup, max_ntries=100, comm=None):
    """
    Sample all cluster positions simultaneously, by making the approximation that 
    the likelihoods for each cluster are independent of one another (i.e. no 
    overlap).
    
    Parameters
    ----------
    
    pos : list of array_like
        List of coords. for each cluster.
    
    cov_prop : array_like
        Covariance matrix for MCMC proposal distribution.
    
    cmbmap : 2D litemap
        Map of the current CMB signal sample
    
    tsz_amps, ksz_amps : array_like
        Current sample of SZ cluster amplitudes. ksz_amps is optional.
    
    cl_set : ClusterSet object
        Collection of clusters.
    
    expt_setup : tuple of lists
        Tuple containing lists of experimental data/settings for each band 
        (data map, inverse noise map, beam template, and band frequency).
        
        The structure of expt_setup is: ([datamap], [Ninv], [beam], [freq1]),
        where each list has length Nbands.
    
    max_ntries : int, optional
        Max. number of rejections to allow for each cluster before giving up 
        and quitting the sampler.
    
    comm : MPI communicator (optional)
        MPI communicator. If not defined, no parallelisation will be used.
    
    Returns
    -------
    new_positions: list of array_like
        List of sampled positions for each cluster.
    """
    datamaps, ninvs, beams, freqs = expt_setup
    g_nu = [cl_set.tsz_spectrum(nu=f) for f in freqs]
    
    # Setup MPI
    rank = None
    if comm is not None: rank = comm.Get_rank()
    my_cluster_idxs = idxs_for_worker(cl_set.Ncl, rank, comm)
    
    # Get profiles for current parameters, for all clusters
    # (FIXME: Could probably reuse these from previous Gibbs step)
    use_ksz = False
    if ksz_amps is not None: use_ksz = True
    params = cl_set.get_all_profile_params()
    tsz_profs, ksz_profs = get_cluster_profiles(params, cl_set, ksz=use_ksz, comm=comm)
    
    # Do MCMC step for each cluster
    new_positions = []
    for i in my_cluster_idxs:
        #print ">>> Worker", rank, "processing cluster", i
        
        # Get current position of cluster
        cur_ra, cur_dec = cl_set.galactic_to_ra_dec(l=pos[i][0], b=pos[i][1])
        
        # Get residual maps and chi-squared for current position
        r = residual_maps_for_cluster( i, tsz_profs, ksz_profs, cmbmap, tsz_amps, 
                                       ksz_amps, cl_set, expt_setup )
        tsz_clmap, ksz_clmap = skymap_for_cluster(i, tsz_profs[i], ksz_profs[i], 
                                                  cl_set, ra=cur_ra, dec=cur_dec)
        chisq0 = 0.
        for k in range(len(freqs)):
            _clmap = tsz_amps[i] * g_nu[k] * tsz_clmap
            if use_ksz:
                _clmap += ksz_amps[i] * ksz_clmap
            _clmap = applyBeam(_clmap, beams[k])
            r0 = r[k] - _clmap
            chisq0 += np.sum(r0 * ninvs[k] * r0)
        
        # Propose new positions until sample is accepted
        accepted = False; ntries = 0
        while not accepted:
        
            # Propose new set of coords and calculate residual, chisq.
            prop = np.random.multivariate_normal(pos[i], cov_prop)
            prop_ra, prop_dec = cl_set.galactic_to_ra_dec(prop[0], prop[1])
            tclmap_prop, kclmap_prop = skymap_for_cluster(i, tsz_profs[i], 
                                 ksz_profs[i], cl_set, ra=prop_ra, dec=prop_dec)
            chisq1 = 0.
            for k in range(len(freqs)):
                _clmap_prop = tsz_amps[i] * g_nu[k] * tclmap_prop
                if use_ksz:
                    _clmap_prop += ksz_amps[i] * kclmap_prop
                _clmap_prop = applyBeam(_clmap_prop, beams[k])
                rp = r[k] - _clmap_prop
                chisq1 += np.sum(rp * ninvs[k] * rp)
            
            #print "\tchisq:", round(chisq1,0), round(chisq0,0), round(chisq1 - chisq0, 4)
            
            # Do Hastings accept/reject test
            u = np.random.uniform()
            mh = np.exp(-0.5*(chisq1 - chisq0))
            if mh >= u:
                print "\t", rank, "ACCEPT: %3.5f" % mh
                accepted = True
            else:
                print "\t", rank, "REJECT: %3.5f" % mh, chisq1, chisq0
            
            # Test to prevent infinite loop if sample can't be returned
            ntries += 1
            if ntries > max_ntries:
                print "ERROR: Failed to accept sample for cluster", i, 
                print "after", ntries, "tries. Quitting."
                comm.Abort(); raise ValueError()
                
        # Save new values of parameters
        new_positions.append(prop)
    
    # Gather updated parameters from all processes
    new_positions = gather_list_from_all(new_positions, cl_set.Ncl, comm)
    
    # Update profile parameters and return new set of parameters
    set_cluster_positions(new_positions, cl_set)
    return new_positions

