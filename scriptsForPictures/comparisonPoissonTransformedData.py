import numpy as np
import matplotlib.pyplot as plot
import vigra

def fac(n):
    erg = 1
    while n!=0:
        erg = erg * n
        n = n - 1
    return erg


def poisson0(x, lamb):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = 1.*np.power(lamb,x[i]) / fac(x[i]) * np.exp(-lamb)
    return y


bild = []
nbrPoints = 3000
for i in range(nbrPoints):
  print i
  bild.append(vigra.impex.readImage('/home/herrmannsdoerfer/DatenBiologenAlsTif/Pos2_2_red2.tif',index = i))

x0=  [50,100,105,200,111,33,33 ,100]
x1 = [50,100,105,200,111,36,200,10 ]

bild2 = np.array(bild)
bild3 = (bild2 - 380)/3.9

x = range(30)

for j in range(len(x0)):
  y = poisson0(x, np.mean(bild3[:,x0[j],x1[j],0]))*nbrPoints
  plot.hist(bild3[:,x0[j],x1[j],0], int(np.max(bild3[:,x0[j],x1[j],0])-np.min(bild3[:,x0[j],x1[j],0])))
  plot.plot(x,y, linewidth = 3)

  #plot.title("Comparison Poisson distribution transformed data", fontsize=20)
  plot.xlabel("bin", fontsize=20)
  plot.ylabel("count per bin", fontsize=20)
  plot.tick_params(axis='both', labelsize=12)

  fname = '/home/herrmannsdoerfer/MasterArbeit/pictures/IsItPoisson'+str(x0[j])+'_'+str(x1[j])+'.png'
  plot.savefig(fname)
  plot.close()