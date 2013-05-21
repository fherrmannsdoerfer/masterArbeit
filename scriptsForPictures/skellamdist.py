import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plot
from scipy import stats

xmax = 30
x1 = np.linspace(1,xmax, xmax)

mu1 = 20
mu2 = 5

y1 = stats.poisson.pmf(x1,mu1)
y2 = stats.poisson.pmf(x1,mu2)

y3 = stats.skellam.pmf(x1,mu1,mu2)


f1 = interp1d(x1,y1,kind='cubic')
f2 = interp1d(x1,y2,kind='cubic')
f3 = interp1d(x1,y3,kind='cubic')

cutoff = 0.001

idx = np.where(y1>cutoff)
y1 = y1[idx]
xy1 = x1[idx]
idx = np.where(y2>cutoff)
y2 = y2[idx]
xy2 = x1[idx]
idx = np.where(y3>cutoff)
y3 = y3[idx]
xy3 = x1[idx]

c1 = [0.,0,.5]
c2 = [0.5,0.,0]
c3 = [0.5,0.,0.5]

plot.scatter(xy1,y1,marker='o', color=c1,label='Poiss mu = '+str(mu1))
plot.scatter(xy2,y2,marker='o', color=c2,label='Poiss mu = '+str(mu2))
plot.scatter(xy3,y3,marker='o', color=c3,label='Skellam mu = '+str(mu1-mu2)+', var = '+str(mu1+mu2))

x2 = np.linspace(1,xmax,600)
plot.plot(x2, f1(x2),color=c1)
plot.plot(x2,f2(x2),color=c2)
plot.plot(x2,f3(x2),color=c3)

plot.axis([1,xmax,-0.01,0.2])

plot.legend()

#a = plot.gca()
#a.set_aspect(40)

#plot.title("Accuracy test for estimated PSF width", fontsize=20)
plot.xlabel("number", fontsize=15)
plot.ylabel("probability", fontsize=15)
plot.tick_params(axis='both', labelsize=12)
plot.legend(loc = 0, prop={'size':12})
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/skellamdist.png', dpi=300)