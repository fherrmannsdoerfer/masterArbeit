import vigra
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats

rg = vigra.impex.readImage('/home/herrmannsdoerfer/MasterArbeit/daten/MAXproj_Pos2_2_green.tif')
rb = vigra.impex.readImage('/home/herrmannsdoerfer/MasterArbeit/daten/MAXProj_Pos2_2_red2.tif')
vigraimgaligned=vigra.RGBImage(np.zeros([289,322,3]))

rgcorr = np.zeros([289,322,1])
tmp = np.zeros([300,400])
for i in range(289):
    for j in range(322):
        tmp[i+3,j+2] = rg[i,j,0]

rgcorr[:,:,0] = tmp[:289,:322]

vigraimgaligned[...,1]=rgcorr[...,0]
vigraimgaligned[...,0]=rb[...,0]

brs = vigra.impex.readImage('/home/herrmannsdoerfer/MasterArbeit/daten/outputColorcomp.png')
#brs = vigra.sampling.resize(np.array(b), shape = [289,322,3])
vigraimg2 = vigra.sampling.resize(np.array(vigraimgaligned), shape = brs.shape)

mmx = stats.scoreatpercentile(brs[np.where(brs>0)].flat, 95.)
if mmx > 0:
    brs[brs>mmx] = mmx # crop maximum at above percentile
    brs=brs/np.max(brs)*255
print mmx
mmx = stats.scoreatpercentile(vigraimg2[np.where(vigraimg2>0)].flat, 99.9)
if mmx > 0:
    vigraimg2[vigraimg2>mmx] = mmx # crop maximum at above percentile
    vigraimg2[...,1]-=750
    vigraimg2[...,0]-=550
    vigraimg2[vigraimg2<0] = 0
    vigraimg2=vigraimg2/(np.max(vigraimg2)-np.min(vigraimg2))*255

print mmx
print vigraimg2.shape, brs.shape[1]/2
vigraimg2[:,:brs.shape[1]/2,:] = brs[:,:brs.shape[1]/2,:]
vigra.impex.writeImage(vigraimg2,'/home/herrmannsdoerfer/MasterArbeit/pictures/alignedStormWidefield.png')