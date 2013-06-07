import numpy as np
import matplotlib.pyplot as plot

dataarr=np.load('/home/herrmannsdoerfer/MasterArbeit/daten/ergebnissBleachingPos2_2_green.npy')
meanar = dataarr[10,:]

plot.plot(range(len(meanar)), meanar)

#plot.title("Bleaching signal", fontsize=20)
plot.xlabel("Frame", fontsize=20)
plot.ylabel("Mean background intensity", fontsize=20)
plot.tick_params(axis='both', labelsize=12)

plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/bleaching.png')