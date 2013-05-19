import numpy as np
import matplotlib.pyplot as plot

datax=np.load('/home/herrmannsdoerfer/MasterArbeit/daten/actuallSigmas.npy')
datay=np.load("/home/herrmannsdoerfer/MasterArbeit/daten/estimatedSigmas.npy")

skip = 6

datax = np.delete(datax,5)
datay = np.delete(datay,5)

plot.plot(datax, datay,label = "measurement", linewidth = 2)
plot.plot(datax, datax,label = "ideal values", linewidth = 2)
plot.title("Accuracy test for estimated PSF width", fontsize=20)
plot.xlabel("true width", fontsize=20)
plot.ylabel("estimated width", fontsize=20)
plot.tick_params(axis='both', labelsize=12)
plot.legend(loc = 0, prop={'size':15})

plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/AccuracyTestPSFWidth.png')