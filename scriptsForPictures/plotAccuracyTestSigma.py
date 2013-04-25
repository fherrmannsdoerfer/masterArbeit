import numpy as np
import matplotlib.pyplot as plot

data0=np.loadtxt("/home/herrmannsdoerfer/MasterArbeit/daten/intensityAccuracyData1spot40_40_pixelsSigma_0_6.txt")
data1=np.loadtxt("/home/herrmannsdoerfer/MasterArbeit/daten/intensityAccuracyData1spot40_40_pixelsSigma_1.txt")
data3=np.loadtxt("/home/herrmannsdoerfer/MasterArbeit/daten/intensityAccuracyData1spot40_40_pixelsSigma_1_4.txt")
data4=np.loadtxt("/home/herrmannsdoerfer/MasterArbeit/daten/intensityAccuracyData1spot40_40_pixelsSigma_1_6.txt")
data5=np.loadtxt("/home/herrmannsdoerfer/MasterArbeit/daten/intensityAccuracyData1spot40_40_pixelsSigma_1_8.txt")
data7=np.loadtxt("/home/herrmannsdoerfer/MasterArbeit/daten/intensityAccuracyData1spot40_40_pixelsSigma_2_2.txt")

skip = 20


plot.plot(data1[skip:,1]/100, data0[skip:,0], label = "sigma = 0.6")
plot.plot(data1[skip:,1]/100, data1[skip:,0], label = "sigma = 1.0")
plot.plot(data1[skip:,1]/100, data3[skip:,0], label = "sigma = 1.4")
plot.plot(data1[skip:,1]/100, data4[skip:,0], label = "sigma = 1.6")
plot.plot(data1[skip:,1]/100, data5[skip:,0], label = "sigma = 1.8")
plot.plot(data1[skip:,1]/100, data7[skip:,0], label = "sigma = 2.2")

plot.title("Accuracy test", fontsize=20)
plot.xlabel("SNR", fontsize=20)
plot.ylabel("Std. dev. in pixel", fontsize=20)
plot.tick_params(axis='both', labelsize=12)

plot.legend(loc = 1, prop={'size':15})
plot.xlim([data1[skip,1]/100, data1[-1,1]/100])
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/AccuracyTestSigma.png')