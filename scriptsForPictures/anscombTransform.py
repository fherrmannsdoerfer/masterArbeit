import numpy as np
import matplotlib.pyplot as plot
import math

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


def poisson(x, lamb):
    y = np.zeros(len(x))
    if lamb > 50:
        lamb = int(lamb)
    if lamb > 100:
        return y
    if len(x)>2:
        y[0] = float(pow(lamb,x[0])) / float(math.factorial(x[0])) * float(np.exp(-lamb))
        y[1] = float(pow(lamb,x[1])) / float(math.factorial(x[1])) * float(np.exp(-lamb))
        for i in range(2,len(x)):
            if i > lamb and y[i-1] < 1e-25:
                return y
            else:
                a = math.log(pow(lamb,x[i]))
                b = math.log(math.factorial(x[i]))
                c = np.exp(-lamb)
                d = a - b
                e = np.exp(d)
                y[i] = e * c
        return y
    else:
        return y



m = np.array(range(2000))/100.


y = []
yorig = []
for i in range(len(m)):
    list_p = []
    for j in range(100000):
        print i, j
        list_p.append(np.random.poisson(m[i]))
    y.append(np.sqrt(np.var(2*np.sqrt(np.array(list_p)+3./8))))
    yorig.append(np.sqrt(np.var(np.array(list_p))))

fst = 30
fsa = 25
fsl = 25
lw = 2
a=plot.plot(m,y, linewidth = lw)
b=plot.plot(m,yorig, linewidth = lw)
#plot.title("Anscomb transformation", fontsize=fst)
plot.xlabel("mean", fontsize = fsa)
plot.ylabel("stddev", fontsize = fsa)

leg = plot.legend((a,b),("Anscomb", "Original"), loc='upper left')
plot.setp(leg.get_texts(),fontsize = fsl)
#plot.setp(plot.gca().label, fontsize = 20)
plot.xticks(fontsize=fsa*2/3.)
plot.yticks(fontsize=fsa*2/3.)
plot.ylim([0,4])
#plot.show()
plot.savefig("/home/herrmannsdoerfer/master/workspace/Master_Arbeit/pictures/anscombe.png")
print "done"