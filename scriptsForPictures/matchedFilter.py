import numpy as np
import matplotlib.pyplot as plot

arrs = []

x = np.array(range(4,40))/10.
color = [[1,0,0],[47/255.,79/255.,47/255.],[0,0,1],[173/255.,1,47/255.],[104/255.,34/255.,139/255.]]
a= -1
for i in range(13,30,4):
    a+=1
    print i,a
    arrs.append(np.load('/home/herrmannsdoerfer/MasterArbeit/daten/DatenMatchedFilter/vargoodfac128sigma'+str(i/10.)+'.npy'))
    plot.plot(x,arrs[-1],linewidth = 3, label = 'psf scale = '+str((i+1)/10.), color = color[a])
    plot.scatter(x[i-3], arrs[-1][i-3], marker = 'x', s = 200,color = color[a], linewidths = 3)

plot.tick_params(axis='both', which='major', labelsize=15)
plot.legend()
#plot.title("Accuracy for different PSFs and filterwidths", fontsize=20)
plot.xlabel("Filterwidth matched filter", fontsize=20)
plot.ylabel("Standard deviation in pixels", fontsize=20)
plot.axis([0.5,3.9,0.1,2])
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/matchedFilterPlots1.png')
plot.close()


i = 25
plot.plot(x,arrs[-2]-.073,linewidth = 3, label = 'shifted measurement', color = color[0])
plot.plot(x,arrs[-2],'--',linewidth = 1, label = 'measurement', color = color[1])
s = 1000.
n = 10.
sigmapsf = 2.6
y = n/s*np.sqrt(np.sqrt(2)*np.pi/2)*(1+sigmapsf**2/x**2)*(sigmapsf**2+x**2)

plot.plot(x,y, label = 'calculation')

plot.tick_params(axis='both', which='major', labelsize=15)
plot.legend()
#plot.title("Comparison between calculated and measured error", fontsize=20)
plot.xlabel("Filterwidth matched filter", fontsize=20)
plot.ylabel("Standard deviation in pixels", fontsize=20)
plot.axis([0.5,3.9,0.3,2])
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/matchedFilterPlots2.png')
plot.close()

i = 17
plot.plot(x,arrs[-4]-.046,linewidth = 3, label = 'shifted measurement', color = color[0])
plot.plot(x,arrs[-4],'--',linewidth = 1, label = 'measurement', color = color[1])
s = 1000.
n = 10.
sigmapsf = 1.8
y = n/s*np.sqrt(np.sqrt(2)*np.pi/2)*(1+sigmapsf**2/x**2)*(sigmapsf**2+x**2)

plot.plot(x,y, label = 'calculation')

plot.tick_params(axis='both', which='major', labelsize=15)
plot.legend()
#plot.title("Comparison between calculated and measured error", fontsize=20)
plot.xlabel("Filterwidth matched filter", fontsize=20)
plot.ylabel("Standard deviation in pixels", fontsize=20)
plot.axis([0.5,3.9,0.1,0.8])
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/matchedFilterPlots3.png')
plot.close()