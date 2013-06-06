import numpy as np
import matplotlib.pyplot as plot
import vigra

def sigma2D(mat):
    [dim0, dim1] = np.array(mat).shape
    var = 0
    for i in range(dim0):
        for j in range(dim1):
            var +=  (((j-int(dim1/2.)))**2+((i-int(dim0/2.)))**2)*mat[i,j]
    return np.sqrt(var/np.sum(mat)/2.)

listansc=[]
listwoansc=[]
listretansc=[]
logspace = np.logspace(-4,8,100)
xvalues = []
for i in logspace:
    print "i: ",i
    sig1 = 1
    int1 = i

    bild1 = np.zeros([100,100])
    bild2 = np.zeros([100,100])
    bild3 = np.zeros([100,100])

    bild1[50,50] = int1



    bild1sm = vigra.gaussianSmoothing(bild1.astype(np.float32),sig1, window_size = 10)

    #plot.matshow(bild1sm)
    #plot.show()

    bild2sm = vigra.gaussianSmoothing(bild1.astype(np.float32),sig1, window_size = 10)
    bild3sm = vigra.gaussianSmoothing(bild1.astype(np.float32),sig1, window_size = 10)

    print bild1sm[50,50]

    bild2sm = 2*np.sqrt(bild2sm+0.375) - 2*np.sqrt(0.375)
    bild3sm = 2*np.sqrt(bild3sm+0.375) - 2*np.sqrt(0.375)
    bild3sm = ((bild3sm+3./8.)/2)**2-3./8.
    print bild2sm[50,50]
    xvalues.append(bild2sm[50,50])
    print "ohne Ansc:",sigma2D(bild1sm)
    print "mit Ansc: ",sigma2D(bild2sm)
    print "mit 2**.25:",sigma2D(bild3sm)
    listwoansc.append(sigma2D(bild1sm))
    listansc.append(sigma2D(bild2sm))
    listretansc.append(sigma2D(bild3sm))


plot.show()
plot.plot(xvalues, listwoansc, label="without Anscombe", linewidth = 2)
plot.plot(xvalues, listansc, label="with Anscombe", linewidth = 2)
#plot.plot(xvalues, listretansc, label="mit ret ansc")
plot.xscale('log')
plot.xlabel("SNR", fontsize=20)
plot.ylabel("Std. dev. in pixel", fontsize=20)
plot.tick_params(axis='both', labelsize=12)

plot.legend(loc = 2, prop={'size':15})
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/AnscombeChangesWidth.png')