import numpy as np
import matplotlib.pyplot as plot

def gauss(x,a,b,c,d):
  return a +(b-a)*np.exp(-0.5*(x-c)*(x-c)/d/d)

ansc = np.loadtxt('/home/herrmannsdoerfer/MasterArbeit/daten/Histogram of AnscombeAndBackgroundCorrectedFrame60.txt', skiprows = 1)
poiss = np.loadtxt('/home/herrmannsdoerfer/MasterArbeit/daten/Histogram of PoissonCorrectedFrame60.txt', skiprows = 1)

a1 = 2.865
b1 = 470.68
c1 = -0.018
d1 = 1.00747

print ansc.shape
print poiss.shape
x = ansc[:150,0]
plot.bar(x,ansc[:150,1], width = ansc[1,0]-ansc[0,0], color='g')
y = gauss(x,a1,b1,c1,d1)
plot.plot(x,y,color = 'k', linewidth = 3)

plot.title("Histogram after Anscombe transformation", fontsize=20)
plot.xlabel("intensity", fontsize=20)
plot.ylabel("pixel per bin", fontsize=20)




#x = np.array(range(int(np.min(meansk)), int(np.max(meansk))) )
#y = 5 * x - 5*380
#plot.plot(x,y, label = "fit = 5.01 * x - 1909", linewidth = 3)

#plot.title("Fitting example", fontsize=20)
#plot.xlabel("mean", fontsize=20)
#plot.ylabel("Skellam parameter", fontsize=20)
#plot.tick_params(axis='both', labelsize=12)


#plot.legend(loc = 1, prop={'size':15})
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/anscombeAndFit.png')

