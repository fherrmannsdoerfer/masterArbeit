import numpy as np
import matplotlib.pyplot as plot

def doSkellam(data):
    diffs=data[1:]-data[:-1]
    meandiff = (diffs[0]-diffs[-1])/np.float(data.shape[0])
    sig2 = np.sum((meandiff-diffs)**2)/(data.shape[0]-1)
    return (meandiff+sig2)/2.

listvarvar = []
listvarskellam = []
listmeanskellam = []
listmeanvar=[]
for i in range(1,40):
    print i
    listvar =[]
    listskellam = []
    for j in range(200000):
        data = np.random.poisson(i,200)
        listvar.append(np.var(data))
        listskellam.append(doSkellam(data))
    listvarvar.append(np.var(listvar))
    listvarskellam.append(np.var(listskellam))
    listmeanskellam.append(np.mean(listskellam))
    listmeanvar.append(np.mean(listvar))

plot.figure(1)

plot.plot(range(1,40), listvarvar, linewidth = 3, label="variance")
plot.plot(range(1,40), listvarskellam, linewidth = 3, label="Skellam")

x0=0
x1=40
y0=0
y1=40
dx = (x1-x0)/.8
dy = (y1-y0)/.8
xklein0=38
xklein1=39
yklein0=37.8
yklein1=39.2
plot.legend(loc = 'upper left', bbox_to_anchor = (0.1, 0.9), prop={'size':15})
plot.xlabel("mean of the simulated Poisson distribution", fontsize=15)
plot.ylabel("variance of the estimated mean", fontsize=15)
plot.tick_params(axis='both', labelsize=12)
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/SkellamVarVar.png', dpi=300)

plot.figure(2)
plot.axes([0.1,0.1,.8,.8])
plot.plot(range(1,40), listmeanvar, linewidth = 3, label="variance")
plot.plot(range(1,40), listmeanskellam, linewidth = 3, label="Skellam")
plot.axis([x0,x1,y0,y1])

plot.plot([xklein0,xklein1,xklein1,xklein0,xklein0],[yklein0,yklein0,yklein1,yklein1,yklein0], color = [0,0,0])
plot.plot([xklein0,(0.2-0.1)*dx],[yklein0,(0.55-0.1)*dy], color = [0,0,0], linestyle = '--')
plot.plot([xklein0,(0.2-0.1)*dx],[yklein1,(0.55+0.3-0.1)*dy], color = [0,0,0], linestyle = '--')
plot.plot([xklein1,(0.2+0.3-0.1)*dx],[yklein0,(0.55-0.1)*dy], color = [0,0,0], linestyle = '--')
plot.plot([xklein1,(0.2+0.3-0.1)*dx],[yklein1,(0.55+0.3-0.1)*dy], color = [0,0,0], linestyle = '--')
plot.xlabel("mean of the simulated Poisson distribution", fontsize=15)
plot.ylabel("mean of the estimated mean", fontsize=15)
plot.tick_params(axis='both', labelsize=12)
plot.legend(loc = 'upper left', bbox_to_anchor = (0.6, 0.4), prop={'size':15})
plot.axes([0.2,0.55,.3,.3])
plot.plot(range(1,40), listmeanvar)
plot.plot(range(1,40), listmeanskellam)
plot.axis([xklein0,xklein1,yklein0,yklein1])

plot.xlabel("mean of the simulated Poisson distribution", fontsize=10)
plot.ylabel("mean of the estimated mean", fontsize=10)
plot.tick_params(axis='both', labelsize=12)
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/SkellamVarMean.png', dpi=300)