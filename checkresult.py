import numpy as np
import pylab as P
import experiment
p = experiment.p

nsims = p['nsims']
nsamp = p['nsamp']

resultDir = 'results'


all_cmb=[]
all_mono=[]
all_tsz=[]
all_ksz=[]


for n in range(nsims):
    
    gibbs_cmb=[]
    gibbs_mono=[]
    gibbs_tsz=[]
    gibbs_ksz=[]
    
    for i in range(nsamp):
        print 'gibbs step:',i
        cmb_amp=np.load('%s/cmb_amp_%03d_%03d.npy'%(resultDir,i,n))
        mono_amp=np.loadtxt('%s/mono_amp_%03d_%03d.dat'%(resultDir,i,n))
        tsz_amp=np.loadtxt('%s/tsz_amp_%03d_%03d.dat'%(resultDir,i,n))
        ksz_amp=np.loadtxt('%s/ksz_amp_%03d_%03d.dat'%(resultDir,i,n))

        gibbs_cmb+=[cmb_amp]
        gibbs_mono+=[mono_amp]
        gibbs_tsz+=[tsz_amp]
        gibbs_ksz+=[ksz_amp]

    mean_cmb=np.mean(gibbs_cmb,axis=0)
    std_cmb=np.std(gibbs_cmb,axis=0)

    mean_mono=np.mean(gibbs_mono,axis=0)
    std_mono=np.std(gibbs_mono,axis=0)

    mean_tsz=np.mean(gibbs_tsz,axis=0)
    std_tsz=np.std(gibbs_tsz,axis=0)

    mean_ksz=np.mean(gibbs_ksz,axis=0)
    std_ksz=np.std(gibbs_ksz,axis=0)

    all_cmb+=[mean_cmb]
    all_mono+=[mean_mono]
    all_tsz+=[mean_tsz]
    all_ksz+=[mean_ksz]


    print 'monopole:',mean_mono,'+/-',std_mono
    print 'tsz:',mean_tsz,'+/-',std_tsz
    print 'ksz:',mean_ksz,'+/-',std_ksz

    P.matshow(mean_cmb)
    P.colorbar()
    P.show()


    P.matshow(std_cmb)
    P.colorbar()
    P.show()


mean_all_cmb=np.mean(all_cmb,axis=0)
std_all_cmb=np.std(all_cmb,axis=0)

mean_all_mono=np.mean(all_mono,axis=0)
std_all_mono=np.std(all_mono,axis=0)

mean_all_tsz=np.mean(all_tsz,axis=0)
std_all_tsz=np.std(all_tsz,axis=0)


P.plot(all_ksz)
P.show()
mean_all_ksz=np.mean(all_ksz,axis=0)
std_all_ksz=np.std(all_ksz,axis=0)

print 'mean monopole:',mean_all_mono,'+/-',std_all_mono
print 'mean tsz:',mean_all_tsz,'+/-',std_all_tsz
print 'mean ksz:',mean_all_ksz,'+/-',std_all_ksz

