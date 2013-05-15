import numpy as np
import matplotlib.pyplot as plot
import vigra
import os
import sys


fname = '/home/herrmannsdoerfer/MasterArbeit/daten/Tubulin2ForPostProc.txt'
outputname = os.path.splitext(fname)[0] + '_advanced' +   os.path.splitext(fname)[1]
factor = 4.

dims = np.loadtxt(fname)[0,:]
data = np.loadtxt(fname, skiprows = 1)
data[:,:2] = data[:,:2]/100.

mat = np.zeros([(dims[0])*factor, (dims[1])*factor])

print mat.shape

for i in range(len(data)):
    mat[data[i,0]*factor, data[i,1] * factor] += 1

matsmoothed = vigra.gaussianSmoothing(np.array(mat, dtype = np.float32), 1.)
thr = (np.max(matsmoothed) - np.min(matsmoothed))/100. * 3.2
matsmoothedThresholded = np.where(matsmoothed < thr, 0, 1)
print thr
print np.mean(matsmoothed)

listDeleted = []
newCoords = []
for i in range(len(data)):
    if matsmoothedThresholded[data[i,0]*factor, data[i,1] * factor] == 1:
        newCoords.append(data[i,:])
    else:
        listDeleted.append(data[i,:])

newCoords = np.array(newCoords)
listDeleted= np.array(listDeleted)
mat2 = np.zeros([(dims[0])*factor, (dims[1])*factor])

for i in range(len(newCoords)):
    mat2[newCoords[i,0]*factor, newCoords[i,1] * factor] += 1

mat3 = np.copy(mat2)

#print len(listDeleted)
#maxVal = np.max(mat2)
#for i in range(len(listDeleted)):
    #for x in range(5):
        #mat2[listDeleted[i,0]*factor+x, listDeleted[i,1]*factor] = maxVal
        #mat2[listDeleted[i,0]*factor-x, listDeleted[i,1]*factor] = maxVal
        #mat2[listDeleted[i,0]*factor, listDeleted[i,1]*factor+x] = maxVal
        #mat2[listDeleted[i,0]*factor, listDeleted[i,1]*factor-x] = maxVal

mat = np.where(mat>0, 0.99*np.max(mat), mat)
mat3 = np.where(mat3>0, 0.99*np.max(mat3), mat3)
#plot.matshow(mat)
#plot.gray()
#plot.matshow(mat2)
#plot.gray()
#plot.matshow(mat3)
#plot.gray()
#plot.matshow(matsmoothedThresholded)
#plot.gray()
#plot.show()


x1 = 1000/2
x2 = 1800/2
y1 = 300/2
y2 = 1100/2

fig = plot.figure()
fig.set_size_inches(x2-x1, y2-y1)
ax = plot.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(np.transpose(matsmoothedThresholded[x1:x2,y1:y2]), cmap = plot.cm.gray)
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/postproc1.png', dpi = 1)

#plot.matshow(np.transpose(matsmoothedThresholded[x1:x2,y1:y2]))
#plot.gray()
#plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/postproc1.png')

fig = plot.figure()
fig.set_size_inches(x2-x1, y2-y1)
ax = plot.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(np.transpose((mat[x1:x2,y1:y2])), cmap = plot.cm.gray)
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/postproc2.png', dpi = 1)

#plot.matshow(np.transpose(mat[x1:x2,y1:y2]))

#plot.gray()
#plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/postproc2.png')
#plot.matshow(np.transpose(mat3[x1:x2,y1:y2]))
#plot.gray()
#plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/postproc3.png')

fig = plot.figure()
fig.set_size_inches(x2-x1, y2-y1)
ax = plot.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(np.transpose((mat3[x1:x2,y1:y2])), cmap = plot.cm.gray)
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/postproc3.png', dpi = 1)
 
