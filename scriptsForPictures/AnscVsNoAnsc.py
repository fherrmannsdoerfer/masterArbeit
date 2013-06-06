import numpy as np
import matplotlib.pyplot as plot


datanoAnsc=np.load('/home/herrmannsdoerfer/MasterArbeit/daten/AnscombeVsNoAnscombe/vargoodAnscfac128sigma1.5.npy')
datayesAnsc=np.load("/home/herrmannsdoerfer/MasterArbeit/daten/AnscombeVsNoAnscombe/vargoodfac128sigma1.5.npy")

datax = np.array(range(4,40))/10.

plot.plot(datax, datanoAnsc,label = "without Anscombe", linewidth = 2)
plot.plot(datax, datayesAnsc,label = "with Anscombe", linewidth = 2)
plot.xlabel("filter width", fontsize=20)
plot.ylabel("Std. dev. in pixel", fontsize=20)
plot.tick_params(axis='both', labelsize=12)
plot.legend(loc = 0, prop={'size':15})

plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/AnscVsNoAnsc.png')