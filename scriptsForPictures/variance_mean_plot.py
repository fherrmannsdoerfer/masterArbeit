#Dieses Skript erzeugt variance ueber mean plots und speichert die Werte fuer Steigung und Nullstelle 
#mit der Information ueber den betrachteten Ausschnitt ab

import numpy as np
import h5py
from matplotlib import pyplot as plot
from scipy import stats

f = h5py.File('/home/herrmannsdoerfer/Daten_Kalibrierung_Messung_3_lightSecondTry.hdf5')

min0 = 100
max0 = 114
min1 = 100
max1 = 114
tag = 'on'
ordnername = 'Shutter_open_emgain_'+ tag
tiffstacks = f[ordnername]
bild = []
for i in range(len(tiffstacks)):
    print i
    if i == 22:
        continue
    bild.append(tiffstacks.values()[i][min0:max0,min1:max1,:])
    
pic_nbrs = 5#len(bild)    
    

x = np.zeros(pic_nbrs)
y = np.zeros(pic_nbrs)
for k in range(pic_nbrs):
    x[k] = np.mean(bild[k])
    y[k] = np.var(bild[k])
m,c = stats.linregress(x,y)[0:2]
xr = range(0,int(np.max(x)),int(np.max(x)/20))
yr = np.array(m)*xr + c
plot.plot(xr,yr)    
plot.plot(x,y,'o')
plot.show()
ns = -c/m
print m
print ns


x = np.zeros([int(max0-min0), int(max1-min1), pic_nbrs])
y = np.zeros([int(max0-min0), int(max1-min1), pic_nbrs])
m = np.zeros([int(max0-min0), int(max1-min1)])  #Steigung
c = np.zeros([int(max0-min0), int(max1-min1)])
ns = np.zeros([int(max0-min0), int(max1-min1)]) #Nullstelle

for i in range(max0 - min0):
    #print i
    for j in range(max1 - min1):
        for k in range(pic_nbrs):
            x[i,j,k] = (np.mean(bild[k][i,j,:]))
            y[i,j,k] = (np.var(bild[k][i,j,:]))
        m[i,j],c[i,j] = stats.linregress(x[i,j],y[i,j])[0:2]
        xr = range(0,int(np.max(x)),int(np.max(x)/20))
        yr = np.array(m[i,j])*xr + c[i,j]
        #plot.plot(xr,yr)    
        plot.plot(x[i,j],y[i,j],'o')
        ns[i,j] = -c[i,j]/m[i,j]
        #print 'Nullstelle: %3.3f, Steigung: %3.3f' %(-c[i,j]/m[i,j], m[i,j]) 
tmpstr = 'variance-mean-plot (%s-%s,%s-%s)' %(str(min0),str(max0),str(min1),str(max1))
tmpstr2 = 'Nullstellen_%s_(%s-%s,%s-%s)'%(ordnername, str(min0),str(max0),str(min1),str(max1))
np.save(tmpstr2, ns)
tmpstr3 = 'Steigungen_%s_(%s-%s,%s-%s)'%(ordnername, str(min0),str(max0),str(min1),str(max1))
np.save(tmpstr3, m)

f.close()

print np.mean(m)
print np.mean(ns)

print np.median(m)
print np.median(ns)

plot.title(tmpstr)
plot.xlabel('mean')
plot.ylabel('variance')
plot.savefig('/home/herrmannsdoerfer/tmpOutput/'+tmpstr+'Kalibrierung_'+tag)
plot.show()
plot.close()

x = range(0, np.max(bild))
y = np.multiply(np.mean(m),x) - np.mean(ns)* np.mean(m)
plot.plot(x,y)
plot.savefig('/home/herrmannsdoerfer/tmpOutput/'+tmpstr+'Kalibrierung_'+tag+'_mean')
plot.close()