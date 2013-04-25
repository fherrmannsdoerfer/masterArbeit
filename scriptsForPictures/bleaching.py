import numpy as np
import vigra
import matplotlib.pyplot as plt

fname = '/home/herrmannsdoerfer/DatenBiologenAlsTif/Pos2_2_green.tif'

stacksize = vigra.impex.numberImages(fname)

xpos = [10,20,30,40,50,60,70,80,90,100]
ypos = [10,20,30,40,50,60,70,80,90,100]

valuexy = []
meanar=[]
for i in range(len(xpos)):
    valuexy.append([])

for i in range(stacksize):
    print i
    tmp = vigra.impex.readImage(fname, index=i)
    for j in range(len(xpos)):
        valuexy[j].append(tmp[xpos[j], ypos[j]])
    meanar.append(np.mean(tmp))

dataarr = np.zeros([11,stacksize])
for i in range(len(xpos)):
    dataarr[i,:] = valuexy[i]
dataarr[10,:] = meanar
np.save("/home/herrmannsdoerfer/tmpOutput/ergebnissBleachingPos2_2_green", dataarr)

plt.plot(range(len(valuexy[0])), valuexy[0])
plt.show()
plt.plot(range(len(meanar)), meanar)
plt.show()