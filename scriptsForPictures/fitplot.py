import numpy as np
import matplotlib.pyplot as plot

meansk = np.load("/home/herrmannsdoerfer/MasterArbeit/daten/meansk_a_5_b_380.npy")
varsk = np.load("/home/herrmannsdoerfer/MasterArbeit/daten/varsk.npy")

plot.scatter(meansk, varsk)
x = np.array(range(int(np.min(meansk)), int(np.max(meansk))) )
y = 5 * x - 5*380
plot.plot(x,y, label = "fit = 5.01 * x - 1909", linewidth = 3)

#plot.title("Fitting example", fontsize=20)
plot.xlabel("mean", fontsize=20)
plot.ylabel("Skellam parameter", fontsize=20)
plot.tick_params(axis='both', labelsize=12)


plot.legend(loc = 1, prop={'size':15})
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/fittingExample.png')
