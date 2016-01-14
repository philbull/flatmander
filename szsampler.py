#!/usr/bin/python
"""
Flat-sky SZ+CMB Gibbs sampler.
"""
import numpy as np
import pylab as P
import fist
import time
from mpi4py import MPI as mpi
import fft


# Sampling settings
SAMPLE_AMPS = True
SAMPLE_SHAPES = False
SAMPLE_POSITIONS = False
SAMPLE_VCOV = False
# Setup RNG and MPI
#np.random.seed(18)
comm = mpi.COMM_WORLD
rank = comm.Get_rank()

# Load experimental and cosmology settings (p, cosmo)
import experiment
p = experiment.p
cosmo = experiment.cosmo
mapDir = experiment.mapDir
nsims = p['nsims']
nsamp = p['nsamp']




# Load data and instrumental specifications
template, power_2d, beams, ninvs, freqs = fist.experiment_settings(p)

# Speed up FFT by measuring
#t=time.time()
fft.rfft(ninvs[0].copy(),axes=[-2,-1],flags=['FFTW_MEASURE'])
template_l = fft.fft(ninvs[0],axes=[-2,-1])
fft.irfft(template_l,axes=[-2,-1],flags=['FFTW_MEASURE'])


#Create a directory for the results of the chain
resultDir = fist.check_dir_exists('results')



for n in range(nsims):
    datamaps=[]
    count=0
    for k in freqs:
        datamaps+=[fist.litemap_from_fits(mapDir+'/data_%03d_%d.fits'%(n,k))]
        mask=fist.litemap_from_fits(mapDir+'/mask_%d.fits'%(k))

        ninvs[count]*=mask.data
        count+=1
    


    expt_setup = (datamaps, ninvs, beams, freqs)

    # Load cluster catalogue and get initial shape parameter/position sets
    cluster_set = fist.ClusterSet(catalogue=p['cluster_cat'], map_template=datamaps[0])
    shape = cluster_set.get_all_profile_params()
    pos = cluster_set.get_all_positions()

    # Define proposal covmat for shape sampler
    # (0:alpha, 1:beta, 2:gamma, 3:c500, 4:P0, 5:M500, 6:r500, 7:z)
    shape_cov = np.zeros((8,8))
    shape_cov[6,6] = (0.0005)**2. # r500

    # Define proposal covmat for position sampler
    # (0:l, 1:b)
    pos_cov = np.zeros((2,2))
    pos_cov[0,0] = pos_cov[1,1] = (0.0002)**2.


    # Prepare linear system for amplitude sampler
    if SAMPLE_AMPS:
        linsys_setup = fist.cg.prep_system(expt_setup, power_2d, cluster_set,
                                       p['rmsArcmin'][0] ,use_ksz=True)

    # Keep track of parameters as MCMC progresses
    chain_tszamp = []
    chain_kszamp = []
    chain_shape = []
    chain_pos = []


    print 'realization %d'%n
    for i in range(nsamp):
        if rank == 0:
            print "\n * Gibbs step", i+1, "/", nsamp
            t0 = time.time()

        # ==========================================================================
        # (1) AMPLITUDE SAMPLER
        # ==========================================================================
        if SAMPLE_AMPS:
            if rank == 0:
                print "    Sampling amplitudes..."
                
                # Construct initial guess for soln. vector, RHS of linear system
                x = fist.cg.computeX0(linsys_setup)
                b = fist.cg.computeB(linsys_setup)
                
                # Solve system using Conjugate Gradient method
                # (Return only cluster amplitude sample as result)
                t = time.time()
                
                cmb_amp, mono_amp, tsz_amp, ksz_amp = \
                    fist.cg.preCondConjugateGradientSolver(b, x, linsys_setup, eps=p['epsilon'], i_max=p['imax'],plotInterval=p['plotInterval'], mapDir=mapDir )
                
                
                np.save('%s/cmb_amp_%03d_%03d.npy'%(resultDir,i,n),cmb_amp)
                np.savetxt('%s/mono_amp_%03d_%03d.dat'%(resultDir,i,n),mono_amp)
                np.savetxt('%s/tsz_amp_%03d_%03d.dat'%(resultDir,i,n),tsz_amp)
                np.savetxt('%s/ksz_amp_%03d_%03d.dat'%(resultDir,i,n),ksz_amp)


                print "\tMonopole:", mono_amp
                print "\tTSZ/KSZ amp.", tsz_amp, ksz_amp
                print "\tSampler took", round(time.time()-t, 3), "sec."
                
                cmb_amp = cmb_amp.flatten() # Prepare to send via MPI
            
            # Distribute results to all processes
            if rank != 0:
                mono_amp = 0
                cmb_amp = np.empty(template.data.size, dtype=template.data.dtype)
                tsz_amp = np.empty(cluster_set.Ncl, dtype=template.data.dtype)
                ksz_amp = np.empty(cluster_set.Ncl, dtype=template.data.dtype)
            comm.Bcast(mono_amp, root=0)
            comm.Bcast(cmb_amp, root=0)
            comm.Bcast(tsz_amp, root=0)
            comm.Bcast(ksz_amp, root=0)
            cmb_amp = np.reshape(cmb_amp, template.data.shape)
            
            # Append results to file
            chain_tszamp.append(tsz_amp)
            chain_kszamp.append(ksz_amp)
            if rank == 0:
                # Monopole
                f = open("chain_monopole.dat", 'a')
                f.write(str(mono_amp) + "\n")
                f.close()
          
            # TSZ/KSZ amps.
            for k in range(cluster_set.Ncl):
                f = open("chain_ksz_%03d.dat"%k, 'a')
                f.write(str(tsz_amp[k]) + "\n")
                f.close()
                f = open("chain_tsz_%03d.dat"%k, 'a')
                f.write(str(tsz_amp[k]) + "\n")
                f.close()
            
            
        # ==========================================================================
        # (2) CLUSTER SHAPE SAMPLER
        # ==========================================================================
        if SAMPLE_SHAPES:
            if rank == 0: print "    Sampling cluster shapes..."
        
            # Sample profile parameters using MCMC
            shape = fist.mh_sample_cluster_shapes( shape, shape_cov, cmb_amp,
                                               tsz_amp, ksz_amp, cluster_set,
                                               expt_setup, comm=comm )
            chain_shape.append(shape)
            print "\tR500:", shape[0][6]
        
            # Append results to file
            if rank == 0:
                for k in range(cluster_set.Ncl):
                    f = open("chain_shape_%03d.dat"%k, 'a')
                    f.write(" ".join(str(_p) for _p in shape[k])+"\n")
                    f.close()
        
            # Update cluster maps
            if rank == 0: print "\tRecomputing cluster maps..."
            beams = linsys_setup[2]
            use_ksz = False
            if vcov is not None: use_ksz=True
            maps = fist.cg.build_cluster_maps(cluster_set, beams, use_ksz=use_ksz)
            linsys_setup = fist.cg.update_system_setup(linsys_setup, maps=maps)
    
    
        # ==========================================================================
        # (3) CLUSTER POSITION SAMPLER
        # ==========================================================================
        if SAMPLE_POSITIONS:
            if rank == 0: print "    Sampling cluster positions..."
        
            # Sample cluster positions using MCMC
            pos = fist.mh_sample_cluster_positions( pos, pos_cov, cmb_amp, tsz_amp,
                                                ksz_amp, cluster_set, expt_setup, 
                                                comm=comm )
            chain_pos.append(pos)
            if rank == 0: print "\tl, b:", pos[0][0], pos[0][1]
        
        # Append results to file
            if rank == 0:
                for k in range(cluster_set.Ncl):
                    f = open("chain_pos_%03d.dat"%k, 'a')
                    f.write(" ".join(str(_p) for _p in pos[k])+"\n")
                    f.close()
        
            # Update cluster maps
            if rank == 0: print "\tRecomputing cluster maps..."
            beams = linsys_setup[2]
            use_ksz = False
            if vcov is not None: use_ksz=True
            maps = fist.cg.build_cluster_maps(cluster_set, beams, use_ksz=use_ksz)
            linsys_setup = fist.cg.update_system_setup(linsys_setup, maps=maps)
        
    
        # ==========================================================================
        # (4) VELOCITY COVMAT SAMPLER
        # ==========================================================================
        if SAMPLE_VCOV:
            if rank == 0: print "    Sampling KSZ velocity covmat..."
            raise NotImplementedError("Velocity covmat sampling is not currently implemented.")
        
            # Update inverse vcov
            if SAMPLE_AMPS:
                ivcov = fist.cg.build_vcov_inverse(vcov)
                linsys_setup = fist.cg.update_system_setup( linsys_setup,
                                                        vcov=vcov, ivcov=ivcov )
    
    
    # ==========================================================================
    # (5) CMB CL SAMPLER
    # ==========================================================================
    
    # Use "TaperMaster" from arXiv:0809.1092?
    
    # Output timing info
    #if rank == 0: print "    Gibbs iter. took", round(time.time() - t0, 1), "sec."

# Clean-up MPI
#comm.Disconnect()
