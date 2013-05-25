import numpy as np
import matplotlib.pyplot as plot
import vigra

c0 = vigra.impex.readImage('/home/herrmannsdoerfer/MasterArbeit/daten/StormResults/Pos11_coloc50nm00.png')
c1 = vigra.impex.readImage('/home/herrmannsdoerfer/MasterArbeit/daten/StormResults/Pos11_coloc50nm01.png')

c0100 = vigra.impex.readImage('/home/herrmannsdoerfer/MasterArbeit/daten/StormResults/Pos11_coloc100nm0.png')
c1100 = vigra.impex.readImage('/home/herrmannsdoerfer/MasterArbeit/daten/StormResults/Pos11_coloc100nm1.png')

origbild = vigra.impex.readImage('/home/herrmannsdoerfer/MasterArbeit/daten/StormResults/Pos11_.png')


if 0:
    x0 = 200
    x1 = 1710
    y0=660
    y1=2060

    c0 = (np.asarray(c0[x0:x1,y0:y1,2]))
    c1 = (np.asarray(c1[x0:x1,y0:y1,2]))
    origbild = (np.asarray(origbild[x0:x1,y0:y1,:]))

    ch1=255*np.ones([c0.shape[0], c0.shape[1],3])
    ch0=255*np.ones([c0.shape[0], c0.shape[1],3])


    idxorigr = np.where(origbild[...,0]==0)
    idxorigg = np.where(origbild[...,1]==0)

    idxc0=np.where(c0>0)
    idxc1=np.where(c1>0)
    print len(idxc0[0])

    ch0[idxc0[0],idxc0[1],:]=0
    ch1[idxc1[0],idxc1[1],:]=0
    ch0[idxc0[0],idxc0[1],0]=c0[idxc0[0],idxc0[1]]
    ch1[idxc1[0],idxc1[1],1]=c1[idxc1[0],idxc1[1]]
    origbild = (np.asarray(origbild[x0:x1,y0:y1,:]))

    origbild[idxorigr[0], idxorigr[1],:] = 255
    origbild[idxorigg[0], idxorigg[1],:] = 255
else:

    x0 = 450
    x1 = 1310
    y0=660
    y1=1100

    c0 = (np.asarray(c0[x0:x1,y0:y1,2]))
    c1 = (np.asarray(c1[x0:x1,y0:y1,2]))
    origbild = (np.asarray(origbild[x0:x1,y0:y1,:]))
    print [c0.shape[0], c0.shape[1],3]
    ch1=255*np.zeros([c0.shape[0], c0.shape[1],3])
    ch0=255*np.zeros([c0.shape[0], c0.shape[1],3])

    idx0 = np.where(c0>0)
    idx1 = np.where(c1>1)



    ch0[...,0]=c0
    ch1[...,1]=c1
    fac = 0.15
    ch0[idx0[0],idx0[1],1] = fac*ch0[idx0[0],idx0[1],0]
    ch0[idx0[0],idx0[1],2] = fac*ch0[idx0[0],idx0[1],0]
    ch1[idx1[0],idx1[1],0] = fac*ch1[idx1[0],idx1[1],1]
    ch1[idx1[0],idx1[1],2] = fac*ch1[idx1[0],idx1[1],1]



    c0100 = (np.asarray(c0100[x0:x1,y0:y1,2]))
    c1100 = (np.asarray(c1100[x0:x1,y0:y1,2]))


    ch1100=255*np.zeros([c0100.shape[0], c0100.shape[1],3])
    ch0100=255*np.zeros([c0100.shape[0], c0100.shape[1],3])

    idx0100 = np.where(c0100>0)
    idx1100 = np.where(c1100>1)



    ch0100[...,0]=c0100
    ch1100[...,1]=c1100
    fac = 0.15
    ch0100[idx0100[0],idx0100[1],1] = fac*ch0100[idx0100[0],idx0100[1],0]
    ch0100[idx0100[0],idx0100[1],2] = fac*ch0100[idx0100[0],idx0100[1],0]
    ch1100[idx1100[0],idx1100[1],0] = fac*ch1100[idx1100[0],idx1100[1],1]
    ch1100[idx1100[0],idx1100[1],2] = fac*ch1100[idx1100[0],idx1100[1],1]

    ch1vig100 = vigra.Image(ch1100.astype(np.float32))
    ch0vig100 = vigra.Image(ch0100.astype(np.float32))



ch1vig = vigra.Image(ch1.astype(np.float32))
ch0vig = vigra.Image(ch0.astype(np.float32))

print ch1vig[1,1,1], ch1[1,1,1]
#plot.matshow(ch1vig[...,0])
#plot.matshow(ch1vig[...,1])
#plot.matshow(ch1vig[...,2])
#plot.show()
vigra.impex.writeImage(origbild, '/home/herrmannsdoerfer/MasterArbeit/pictures/colocalization/cropedOrig.png')
vigra.impex.writeImage(ch1vig, '/home/herrmannsdoerfer/MasterArbeit/pictures/colocalization/cropedCh1.png')
vigra.impex.writeImage(ch0vig, '/home/herrmannsdoerfer/MasterArbeit/pictures/colocalization/cropedCh0.png')

vigra.impex.writeImage(ch1vig100, '/home/herrmannsdoerfer/MasterArbeit/pictures/colocalization/cropedCh1100.png')
vigra.impex.writeImage(ch0vig100, '/home/herrmannsdoerfer/MasterArbeit/pictures/colocalization/cropedCh0100.png')