import numpy as np
import matplotlib.pyplot as plot
import h5py
import vigra

fnamehdf5 = "/home/herrmannsdoerfer/Challenge/TrainingsDaten/BundledTubesHighDensity/BundledTubesHighDensity.hdf5"
fnameCoordsMyStorm0 = "/home/herrmannsdoerfer/MasterArbeit/daten/BundledTubesHighDensity_sigma_0_01.txt"
fnameCoordsMyStorm11 = "/home/herrmannsdoerfer/MasterArbeit/daten/BundledTubesHighDensity_sigma_1_1.txt"
fnameCoordsSimpleStorm = "/home/herrmannsdoerfer/MasterArbeit/daten/BundledTubesHighDensityThr_15.txt"
fnameGroundtruth = "/home/herrmannsdoerfer/Challenge/TrainingsDaten/BundledTubesHighDensity/GroundTruth/activation.csv"

dataStorm0 = np.loadtxt(fnameCoordsMyStorm0,skiprows=1)
dataStorm11 = np.loadtxt(fnameCoordsMyStorm11,skiprows=1)
dataStormSimpleStorm = np.loadtxt(fnameCoordsSimpleStorm,skiprows=1)
dataGroundtruth = np.loadtxt(fnameGroundtruth, skiprows=1, delimiter = ',')

factor = 8
dataStorm0[:,:2] /=100
dataStorm11[:,:2] /= 100
dataGroundtruth[:,2:4] /= 100
dataGroundtruth[:,1] -=1

print dataStorm0[1,:]
print dataStorm11[1,:]
print dataStormSimpleStorm[1,:]
print dataGroundtruth[1,:]

f = h5py.File(fnamehdf5)
hdf5bild = f["data"]

print hdf5bild.shape
for frame in range(360):#  frame = 80
  bild = np.zeros([hdf5bild.shape[1],hdf5bild.shape[2],3])
  for i in range(3):
    bild[...,i] = np.transpose(hdf5bild[frame,:,:])

  bildupscaled = vigra.sampling.resampleImage(bild.astype(np.float32), 8)
  DetStorm0Frame = np.where(dataStorm0[:,2] == frame)[0]
  DetStorm11Frame = np.where(dataStorm11[:,2] == frame)[0]
  DetStormSimpleStormFrame = np.where(dataStormSimpleStorm[:,2] == frame)[0]
  DetGroundTruth = np.where(dataGroundtruth[:,1] == frame)[0]

  print DetStorm0Frame
  print len(DetStorm0Frame)
  print DetStorm11Frame
  print DetStormSimpleStormFrame

  maxval = np.max(bildupscaled)*2

  for i in range(len(DetStorm0Frame)):
    posx = dataStorm0[DetStorm0Frame[i],0] * factor + factor/2
    posy = dataStorm0[DetStorm0Frame[i],1] * factor + factor/2
    for i in range(-4,5):
      bildupscaled[posx+i,posy,0] = maxval
      bildupscaled[posx,posy+i,0] = maxval

  for i in range(len(DetStorm11Frame)):
    posx = dataStorm11[DetStorm11Frame[i],0] * factor + factor/2
    posy = dataStorm11[DetStorm11Frame[i],1] * factor + factor/2
    for i in range(-4,5):
      bildupscaled[posx+i,posy,1] = maxval
      bildupscaled[posx,posy+i,1] = maxval

  for i in range(len(DetStormSimpleStormFrame)):
    posx = dataStormSimpleStorm[DetStormSimpleStormFrame[i],0] * factor + factor/2
    posy = dataStormSimpleStorm[DetStormSimpleStormFrame[i],1] * factor + factor/2
    for i in range(-4,5):
      bildupscaled[posx+i,posy,2] = maxval
      bildupscaled[posx,posy+i,2] = maxval

  for i in range(len(DetGroundTruth)):
    posx = dataGroundtruth[DetGroundTruth[i],2] * factor# + factor/2
    posy = dataGroundtruth[DetGroundTruth[i],3] * factor# + factor/2
    for i in range(-4,5):
      bildupscaled[posx+i,posy+i,:] = maxval
      bildupscaled[posx-i,posy+i,:] = maxval

  bildupscaled[5:10,5:10,0] = maxval
  bildupscaled[15:20,5:10,[0,1]] = maxval
  bildupscaled[25:30,5:10,[0,2]] = maxval

  bildupscaled[5:10,15:20,[0,1]] = maxval
  bildupscaled[15:20,15:20,1] = maxval
  bildupscaled[25:30,15:20,[1,2]] = maxval

  bildupscaled[5:10,25:30,[0,2]] = maxval
  bildupscaled[15:20,25:30,[1,2]] = maxval
  bildupscaled[25:30,25:30,2] = maxval

  vigra.impex.writeImage(bildupscaled, "/home/herrmannsdoerfer/vigoutputtest"+str(frame)+".png")
  #plot.matshow(bildupscaled)
  #plot.show()


f.close()