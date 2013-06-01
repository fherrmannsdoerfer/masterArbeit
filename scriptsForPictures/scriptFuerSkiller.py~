import numpy as np
import matplotlib.pyplot as plot
import h5py
from scipy import stats

f = h5py.File('/home/fherrman/Data/Daten_Kalibrierung_Messung_2.hdf5')
scg4 = f["offene_Blende_gain_4"].values()
scg200 = f["offene_Blende_gain_200"].values()
means4 = []
vars4 = []
means200 = []
vars200 = []
for i in range(len(scg4)):
    means4.append(np.mean(scg4[i]))
    vars4.append(np.var(scg4[i]))
    means200.append(np.mean(scg200[i]))
    vars200.append(np.var(scg200[i]))
    print i

means42 = np.array(means4)[np.array([3,5,7,9,11,13,15,17,19,21,23,25,27])]
vars42 = np.array(vars4)[np.array([3,5,7,9,11,13,15,17,19,21,23,25,27])]
means2002 = np.array(means200)[np.array([3,5,7,9,11,13,15,17,19,21,23,25,27])]
vars2002 = np.array(vars200)[np.array([3,5,7,9,11,13,15,17,19,21,23,25,27])]

plot.figure(1)
#plot.plot(means42,vars42)
m,c = stats.linregress(means42,vars42)[0:2]
xr = np.linspace(np.min(means42), np.max(means42),20)
#xr = range(int(np.min(means42)),int(np.max(means42)),int(np.max(x)/20))
yr = np.array(m)*xr + c
plot.plot(xr,yr)
plot.plot(means42,vars42,'o')
print m,c,"gain4"
plot.savefig('/home/fherrman/pictures/meansVariancesShutteropenGain4.png', dpi=300)

plot.figure(2)
#plot.plot(means2002, vars2002)
m,c = stats.linregress(means2002,vars2002)[0:2]
xr = np.linspace(np.min(means2002), np.max(means2002),20)
#xr = range(int(np.min(means2002)),int(np.max(means2002)),int(np.max(x)/20))
yr = np.array(m)*xr + c
plot.plot(xr,yr)
plot.plot(means2002,vars2002,'o')

print m,c,"gain200"
plot.savefig('/home/fherrman/pictures/meansVariancesShutteropenGain200.png', dpi=300)





f.close()
f = h5py.File('/home/fherrman/Data/Daten_Kalibrierung_Messung_2.hdf5')
scg4 = f["geschlossene_Blende_gain_4"].values()
scg200 = f["geschlossene_Blende_gain_200"].values()
means4 = []
vars4 = []
means200 = []
vars200 = []
for i in range(len(scg4)):
    means4.append(np.mean(scg4[i]))
    vars4.append(np.var(scg4[i]))
    means200.append(np.mean(scg200[i]))
    vars200.append(np.var(scg200[i]))
    print i

means42 = np.array(means4)[np.array([3,5,7,9,11,13,15,17,19,21,23,25,27])]
vars42 = np.array(vars4)[np.array([3,5,7,9,11,13,15,17,19,21,23,25,27])]
means2002 = np.array(means200)[np.array([3,5,7,9,11,13,15,17,19,21,23,25,27])]
vars2002 = np.array(vars200)[np.array([3,5,7,9,11,13,15,17,19,21,23,25,27])]


plot.plot(means42,vars42)
plot.xlabel("mean intensity", fontsize=15)
plot.ylabel("variance", fontsize=15)
plot.tick_params(axis='both', labelsize=12)
#plot.legend(loc = 0, prop={'size':10})
plot.savefig('/home/fherrman/pictures/meansVariancesShutterClosedGain4.png', dpi=300)

plot.plot(means2002,vars2002)
plot.xlabel("mean intensity", fontsize=15)
plot.ylabel("variance", fontsize=15)
plot.tick_params(axis='both', labelsize=12)
#plot.legend(loc = 0, prop={'size':10})
plot.savefig('/home/fherrman/pictures/meansVariancesShutterClosedGain200.png', dpi=300)

f.close()

f = h5py.File('/home/fherrman/Data/Daten_Kalibrierung_Messung_2.hdf5')
scemon = f2["Shutter_closed_emgain_on"].values()
scemoff = f2["Shutter_closed_emgain_off"].values()

listvaroff=[]
listvaron=[]
listmeanon=[]
listmeanoff=[]
for i in range(6):
    listvaroff.append(np.var(scemoff[i]))
    listvaron.append(np.var(scemon[i]))
    listmeanoff.append(np.mean(scemoff[i]))
    listmeanon.append(np.mean(scemon[i]))

exposuretime = [0,200,400,600,800,1000]
plot.plot(exposuretime, listvaron)
plot.plot(exposuretime, listvaroff)

plot.xlabel("exposure time in ms", fontsize=15)
plot.ylabel("variance of intensities", fontsize=15)
plot.tick_params(axis='both', labelsize=12)
#plot.legend(loc = 0, prop={'size':10})
plot.savefig('/home/fherrman/pictures/emgainonoffvariances.png', dpi=300)


plot.plot(exposuretime, listmeanon)
plot.plot(exposuretime, listmeanoff)

plot.xlabel("exposure time in ms", fontsize=15)
plot.ylabel("mean intensities", fontsize=15)
plot.tick_params(axis='both', labelsize=12)
#plot.legend(loc = 0, prop={'size':10})
plot.savefig('/home/fherrman/pictures/emgainonoffmeans.png', dpi=300)