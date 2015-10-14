#!/usr/bin/python
"""
Generate a simulated CMB + SZ cluster sky.
"""
import numpy as np
import fist
import pylab

# Load experimental and cosmology settings (p, cosmo)
import experiment
p = experiment.p
cosmo = experiment.cosmo

# (For testing)
RESCALE_CLUSTER_SIZE = 1. # Enlarge cluster angular size by this factor
RESCALE_TSZ_AMP = 10. # Scale cluster TSZ signal amplitude by this factor
nsims = p['nsims'] # number of CMB simulations

# Make directories for maps, results etc.
mapDir = fist.check_dir_exists(experiment.mapDir)
clusterDir = fist.check_dir_exists('sims/cluster')

# Get experimental settings
template, power_2d, beams, ninvs, freqs = fist.experiment_settings(p)

mask=p['mask']



masks=[]
print mask

for k in freqs:
    
    if mask['apply']:
        _mask=fist.makeMask(template, mask['nHoles'], mask['holeSize'], mask['LenApodMask'],mask['out_pix'])
    else:
        _mask=template.copy()
        _mask.data[:]=1
    _mask.writeFits(mapDir+'/mask_%d.fits'%(k), overWrite=True)
    _mask.plot()
    masks.append(_mask)


for n in range(nsims):
    print "Generating simulation number %03d"%n
    # Generate a realisation of the CMB
    cmbmap = np.random.randn(template.Ny, template.Nx)
    a_l = np.fft.fft2(cmbmap)
    a_l *= np.sqrt(power_2d)
    cmbmap = np.real(np.fft.ifft2(a_l))
    template.data[:] = cmbmap

    # Save CMB map to file
    template.writeFits(mapDir+'/cmbsim_%03d.fits'%n, overWrite=True)

    # Plot result and save to PNG
    fist.plot(template.data, mapDir+'/InitialCMB_%03d.png'%n,'initial CMB', range=(-250., 250.))

    # Load cluster catalogue and generate cluster map
    cluster_set = fist.ClusterSet(catalogue=p['cluster_cat'], map_template=template)
    g_nu = [cluster_set.tsz_spectrum(nu=f) for f in freqs]

    clumap = np.zeros(template.data.shape)
    for i in range(cluster_set.Ncl):
        clumap += cluster_set.get_cluster_map(i, rescale=RESCALE_CLUSTER_SIZE, maptype='tsz')[0]

    # Create skymap per band
    datamaps = []
    count=0
    for k in freqs:
        print " * Simulating band", k, "(%03d GHz)" % k
    
        # Add amplitude-scaled cluster to map (and then plot)
        m = template.copy()
        m.data += g_nu[count] * RESCALE_TSZ_AMP * clumap
        if count == 0 and n==0:
            fist.plot(RESCALE_TSZ_AMP * clumap,'clusters map', mapDir+'/clumap.png')

        # Convolve sky map with the beam
        m = fist.applyBeam(m, beams[count])

        # Add noise to map, and multiply by mask
        
        noise=np.random.randn(m.Ny,m.Nx)*ninvs[count]**(-0.5)
        pylab.matshow(noise)
        pylab.show()
        
        m.data += noise

 

        m.data[:] *= masks[count].data[:]
    
        # Save datamap to FITS file, and make plots
        m.writeFits(mapDir+'/data_%03d_%d.fits'%(n,k), overWrite=True)
        T_plot = m.data[:] * (1 + np.log(masks[count].data[:])) # Useful for seeing masked regions
        fist.plot(T_plot, mapDir+'/data_%03d_%d'%(n,k)+".png",'data %d GHz'%k, range=(-250, 250))
        count+=1

print "Finished."
