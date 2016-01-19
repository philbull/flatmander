
from .utils import *
from .map_tools import *
from .cluster_set import *

def prep_multiple_realisations(p, rescale_cluster_size=1.):
    """
    Prepare templates, cluster maps etc. that will be used when generating 
    multiple realisations of a map.
    """
    # Make directories for maps, results etc.
    mapDir = check_dir_exists('sims/map')
    clusterDir = check_dir_exists('sims/cluster')

    # Get experimental settings
    expt_settings = experiment_settings(p)
    template, power_2d, beams, ninvs, masks, freqs = expt_settings
    
    # Load cluster catalogue and generate cluster map
    cluster_set = ClusterSet(catalogue=p['clusterFile'], map_template=template)
    g_nu = [cluster_set.tsz_spectrum(nu=f) for f in freqs]

    clumap = np.zeros(template.data.shape)
    for i in range(cluster_set.Ncl):
        clumap += cluster_set.get_cluster_map(i, rescale=rescale_cluster_size)
    
    return expt_settings, cluster_set, clumap, g_nu

def new_realisation(p, expt_settings, clumap, g_nu, rescale_tsz_amp=1.):
    """
    Generate a new realisation of the CMB and output maps, including clusters.
    """
    template, power_2d, beams, ninvs, masks, freqs = expt_settings
    
    # Generate a realisation of the CMB
    cmbmap = np.random.randn(template.Ny, template.Nx)
    a_l = np.fft.fft2(cmbmap)
    a_l *= np.sqrt(power_2d)
    cmbmap = np.real(np.fft.ifft2(a_l))
    template.data[:] = cmbmap
    
    # Create skymap per band
    datamaps = []
    for k in range(len(freqs)):
        # Add amplitude-scaled cluster to map (and then plot)
        m = template.copy()
        m.data += g_nu[k] * rescale_tsz_amp * clumap

        # Convolve sky map with the beam
        m = applyBeam(m, beams[k])

        # Add noise to map, and multiply by mask
        m = addWhiteNoise(m, p['rmsArcmin'][k])
        m.data[:] *= masks[k].data[:]
        ninvs *= masks[k].data[:]
        
        # Save in array
        datamaps.append(m)
    return datamaps


def generate_cluster():
    """
    TODO
    """
    # (For testing)
    RESCALE_CLUSTER_SIZE = 1. # Enlarge cluster angular size by this factor
    RESCALE_TSZ_AMP = 5. # Scale cluster TSZ signal amplitude by this factor
    nsim = 1 # ID for this simulation

    # Make directories for maps, results etc.
    mapDir = fist.check_dir_exists('sims/map')
    clusterDir = fist.check_dir_exists('sims/cluster')

    # Get experimental settings
    template, power_2d, beams, ninvs, masks, freqs = experiment_settings(p)

    # Generate a realisation of the CMB
    cmbmap = np.random.randn(template.Ny, template.Nx)
    a_l = np.fft.fft2(cmbmap)
    a_l *= np.sqrt(power_2d)
    cmbmap = np.real(np.fft.ifft2(a_l))
    template.data[:] = cmbmap

    # Save CMB map to file
    template.writeFits(mapDir+'/cmbsim.fits', overWrite=True)

    # Plot result and save to PNG
    fist.plot(template.data, mapDir+'/InitialCMB_%03d.png'%nsim, range=(-300., 300.))

    # Load cluster catalogue and generate cluster map
    cluster_set = fist.ClusterSet(catalogue=p['clusterFile'], map_template=template)
    g_nu = [cluster_set.tsz_spectrum(nu=f) for f in freqs]

    clumap = np.zeros(template.data.shape)
    for i in range(cluster_set.Ncl):
        clumap += cluster_set.get_cluster_map(i, rescale=RESCALE_CLUSTER_SIZE)
    
    # Create skymap per band
    datamaps = []
    for k in range(len(freqs)):
        print " * Simulating band", k, "(%03d GHz)" % freqs[k]    
        
        # Add amplitude-scaled cluster to map (and then plot)
        m = template.copy()
        m.data += g_nu[k] * RESCALE_TSZ_AMP * clumap
        if k == 0:
          fist.plot(RESCALE_TSZ_AMP * clumap, mapDir+'/clumap_%03d.png'%nsim)

        # Convolve sky map with the beam
        m = fist.applyBeam(m, beams[k])

        # Add noise to map, and multiply by mask
        m = fist.addWhiteNoise(m, p['rmsArcmin'][k])
        m.data[:] *= masks[k].data[:]
        ninvs *= masks[k].data[:]
        
        # Save datamap to FITS file, and make plots
        m.writeFits(mapDir+'/data_%02d.fits'%k, overWrite=True)
        T_plot = m.data[:] * (1 + np.log(masks[k].data[:])) # Useful for seeing masked regions
        fist.plot(T_plot, mapDir+'/data_%02d.png'%k, range=(-300., 300.))

    print "Finished."
