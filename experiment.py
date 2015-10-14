
mapDir = "sims/map"

p = {
    'makeTemp':{'apply': True,'raSizeDeg': 3, 'decSizeDeg': 3, 'pixScaleXarcmin':  0.5,'pixScaleYarcmin':  0.5},
    'rmsArcmin':        [8.,7.,25.],
    'beamSizeArcmin':   [2.2,1.4,0.9],
    'freq':             [90,150,230],
    'datamaps':         [mapDir+"/data_00.fits", mapDir+"/data_01.fits", mapDir+"/data_02.fits"],
    #'beamFile':         'data/beamUnity.dat',
    'theoryFile':       'data/bode_almost_wmap5_lmax_1e4_lensedCls.dat',
    'cluster_cat':      'data/MCXC_1clu.dat',
    'ReaStart':         0,
    'ReaStop':          1,
    'epsilon':          1e-7,
    'plotInterval':     1,
    'imax':             3000,
    'buffer':           2, # Increase 
    'out_pix':          0,
    'mask':             {'apply': True, 'nHoles':3, 'holeSize':10, 'LenApodMask':0,'out_pix':0},
    'binningFile':      'data/binningTest',
    'nsims': 1
}

cosmo = {
    'omega_M_0': 		0.3,
    'omega_lambda_0': 	0.7,
    'omega_b_0': 		0.045,
    'omega_n_0':		0.0,
    'omega_k_0':		0.0,
    'N_nu':			    0,
    'h':				0.7,
    'n':				0.96,
    'sigma_8':			0.8,
    'baryonic_effects': True
}

