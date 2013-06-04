import numpy as np
import matplotlib.pyplot as plot
import scipy.stats as stats

nbrPxls = 400

poissDists=[]
listskewest = []
listvar =[]
listmean = []
counteri = 0
for i in range(100,1001,100):
    poissDists.append([])
    listskewest.append([])
    listvar.append([])
    listmean.append([])
    counterj=0
    print counteri
    for j in range(1,100):
        poissDists[counteri].append([])
        tmpskew=[]
        tmpvar=[]
        tmpmean=[]
        for k in range(nbrPxls):
            poissDists[counteri][counterj].append(np.random.poisson(j,i))
            p = poissDists[counteri][counterj][k]
            #print i, len(p),j,(1/stats.skew(p))**2, np.var(p), np.mean(p)
            if np.isnan(1/stats.skew(p)):
                tmpskew.append(0)
            else:
                tmpskew.append(1/stats.skew(p)**2)
            tmpvar.append(np.var(p))
            tmpmean.append(np.mean(p))
        listskewest[counteri].append(np.median(tmpskew))
        listvar[counteri].append(np.median(tmpvar))
        listmean[counteri].append(np.median(tmpmean))
        counterj+=1
    counteri+=1
#skewest2 nbrSamples range(100,1001,100) instead of range(1000,10001,1000)
#skewest3 mean range(1,100) instead of range(1,40)
np.save('/home/herrmannsdoerfer/MasterArbeit/daten/skewest3'+str(nbrPxls), np.array(listskewest))
np.save('/home/herrmannsdoerfer/MasterArbeit/daten/meanest3'+str(nbrPxls), np.array(listmean))
np.save('/home/herrmannsdoerfer/MasterArbeit/daten/varest3'+str(nbrPxls), np.array(listvar))