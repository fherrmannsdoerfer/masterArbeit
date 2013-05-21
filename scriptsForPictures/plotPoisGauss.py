import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plot
from scipy import stats

x1 = np.linspace(1,40, 40)
y2 = stats.poisson.pmf(x1,5)
y4 = stats.poisson.pmf(x1,30)
y1 = stats.poisson.pmf(x1,2)
y3 = stats.poisson.pmf(x1,14)

f1 = interp1d(x1,y1,kind='cubic')
f2 = interp1d(x1,y2,kind='cubic')
f3 = interp1d(x1,y3,kind='cubic')
f4 = interp1d(x1,y4,kind='cubic')

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
idx = np.where(y4>cutoff)
y4 = y4[idx]
xy4 = x1[idx]

c1 = [0.,0,.7]
c2 = [0.1,.2,.8]
c3 = [0.2,.4,.9]
c4 = [0.3,.6,1.0]

plot.scatter(xy1,y1,marker='o', color=c1,label='lambda = 2')
plot.scatter(xy2,y2,marker='o', color=c2,label='lambda = 5')
plot.scatter(xy3,y3,marker='o', color=c3,label='lambda = 14')
plot.scatter(xy4,y4,marker='o', color=c4,label='lambda = 30')
x2 = np.linspace(1,40,600)
plot.plot(x2, f1(x2),color=c1)
plot.plot(x2,f2(x2),color=c2)
plot.plot(x2,f3(x2),color=c3)
plot.plot(x2,f4(x2),color=c4)



x1 = np.linspace(1,40, 40)
y2 = stats.norm.pdf(x1,4.5,np.sqrt(5))
y4 = stats.norm.pdf(x1,29.5,np.sqrt(30))
y1 = stats.norm.pdf(x1,1.5,np.sqrt(2))
y3 = stats.norm.pdf(x1,13.5,np.sqrt(14))

f1 = interp1d(x1,y1,kind='cubic')
f2 = interp1d(x1,y2,kind='cubic')
f3 = interp1d(x1,y3,kind='cubic')
f4 = interp1d(x1,y4,kind='cubic')

idx = np.where(y1>cutoff)
y1 = y1[idx]
xy1 = x1[idx]
idx = np.where(y2>cutoff)
y2 = y2[idx]
xy2 = x1[idx]
idx = np.where(y3>cutoff)
y3 = y3[idx]
xy3 = x1[idx]
idx = np.where(y4>cutoff)
y4 = y4[idx]
xy4 = x1[idx]

c1 = [0.7,.0,0]
c2 = [0.8,.2,0.1]
c3 = [0.9,.4,0.2]
c4 = [  1,.6,0.3]

plot.scatter(xy1,y1,marker='o', color=c1,label='mu = 1.5 ,var = 2')
plot.scatter(xy2,y2,marker='o', color=c2,label='mu = 4.5 ,var = 5')
plot.scatter(xy3,y3,marker='o', color=c3,label='mu = 13.5 ,var = 14')
plot.scatter(xy4,y4,marker='o', color=c4,label='mu = 29.5 ,var = 30')
x2 = np.linspace(1,40,600)
plot.plot(x2, f1(x2),color=c1)
plot.plot(x2,f2(x2),color=c2)
plot.plot(x2,f3(x2),color=c3)
plot.plot(x2,f4(x2),color=c4)
plot.axis([1,40,-0.01,0.35])

plot.legend()

a = plot.gca()
a.set_aspect(50)

#plot.title("Accuracy test for estimated PSF width", fontsize=20)
plot.xlabel("number", fontsize=15)
plot.ylabel("probability", fontsize=15)
plot.tick_params(axis='both', labelsize=12)
plot.legend(loc = 0, prop={'size':10})
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/poissgaussdistr.png', dpi=300)