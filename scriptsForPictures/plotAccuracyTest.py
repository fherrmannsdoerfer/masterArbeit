import numpy as np
import matplotlib.pyplot as plot

data1=np.loadtxt("/home/herrmannsdoerfer/MasterArbeit/daten/IntensityAccuracyData1spot40_40Pixels.txt")
data3=np.loadtxt("/home/herrmannsdoerfer/MasterArbeit/daten/IntensityAccuracyData3spots40_40Pixels.txt")
data5=np.loadtxt("/home/herrmannsdoerfer/MasterArbeit/daten/IntensityAccuracyData5spots40_40Pixels.txt")

skip = 20



plot.plot(data1[skip:,1]/100, data1[skip:,0], label = "1 spot per frame")
plot.plot(data1[skip:,1]/100, data3[skip:,0], label = "3 spot per frame")
plot.plot(data1[skip:,1]/100, data5[skip:,0], label = "5 spot per frame")

#plot.title("Accuracy test", fontsize=20)
plot.xlabel("SNR", fontsize=20)
plot.ylabel("Std. dev. in pixel", fontsize=20)
plot.tick_params(axis='both', labelsize=12)

plot.legend(loc = 1, prop={'size':15})
plot.xlim([data1[skip,1]/100, data1[-1,1]/100])
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/AccuracyTest.png')