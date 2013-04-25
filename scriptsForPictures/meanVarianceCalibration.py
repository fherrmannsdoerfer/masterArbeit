import numpy as np
import matplotlib.pyplot as plot
from scipy import stats
import h5py

a = h5py.File('/home/herrmannsdoerfer/Daten_Kalibrierung_Messung_3_lightSecondTry.hdf5')
data = a["Shutter_open_emgain_on"]

vals=[]

for i in range(11):
    vals.append(data.values()[i])

ms = []
vs = []

for i in range(11):
    ms.append(np.mean(vals[i][:10,:10,:]))
    vs.append(np.var(vals[i][:10,:10,:]))

plot.title("Mean-variance plot of calibration data", fontsize=20)
plot.xlabel("mean", fontsize=20)
plot.ylabel("variance", fontsize=20)
plot.tick_params(axis='both', labelsize=12)

m,c = stats.linregress(ms[:6],vs[:6])[0:2]
xr = np.linspace(np.min(ms),int(np.max(ms)),11)
yr = np.array(m)*xr + c
print m, -c/m
 
plot.plot(ms[:6],vs[:6],label="calibration measurement", linewidth = 3)
plot.plot(xr[:7],yr[:7],label="fitted curve", linewidth = 3)
plot.legend(loc = 2, prop={'size':15})


plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/meanVariancePlotCalibration.png')