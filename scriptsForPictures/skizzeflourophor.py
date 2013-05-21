import numpy as np
from mayavi import mlab

Deltax = 20
Deltay = 0
Deltaz = 5
[phi,theta] = np.mgrid[0:2*np.pi+0.0001:np.pi/200., 0:np.pi+0.0001:np.pi/100.]
R=10
x = R*np.cos(phi)*np.sin(theta)+Deltax
y = R*np.sin(phi)*np.sin(theta)+0
z = R*np.cos(theta)+Deltaz

s = mlab.mesh(x,y,z, opacity = 0.3, color=(0.5,0,0))

[x1,y1] = np.mgrid[0:41:1,0:21:1]
z1 = np.zeros(y1.shape)
z11 = np.ones(y1.shape)*Deltaz

[x2,z2] = np.mgrid[0:41,0:6]
y2=np.zeros(x2.shape)
y22=np.ones(x2.shape)*Deltax
Color = (1,1,1)
s2 = mlab.mesh(x1,y1,z1, color=Color)
s3 = mlab.mesh(x1,y1,z11, color=Color)
s4 = mlab.mesh(x2,y2,z2, color=Color)
s5 = mlab.mesh(x2,y22,z2, color=Color)

origP=[Deltax,Deltay,Deltaz]

phi1 = -100/360.*2*np.pi
theta1 = 50/360.*2*np.pi
orig2 = [Deltax+R*np.cos(phi1)*np.sin(theta1),Deltay+R*np.sin(phi1)*np.sin(theta1),Deltaz+R*np.cos(theta1)]

phi1 = 30/360.*2*np.pi
theta1 = 50/360.*2*np.pi
orig3 = [Deltax+R*np.cos(phi1)*np.sin(theta1),Deltay+R*np.sin(phi1)*np.sin(theta1),Deltaz+R*np.cos(theta1)]

mlab.plot3d([origP[0],orig2[0]],[origP[1],orig2[1]],[origP[2],orig2[2]], tube_radius = .1, tube_sides = 100, color=(1,0,0))
mlab.plot3d([origP[0],orig3[0]],[origP[1],orig3[1]],[origP[2],orig3[2]], tube_radius = .1, tube_sides = 100, color=(0,1,0))


mlab.show()

