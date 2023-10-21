import numpy as np
import matplotlib.pyplot as plt
lenth=21
if lenth%2==0:
    lenth-=1
xa=np.arange(0,lenth)
ya=np.arange(0,lenth)
xv,yv=np.meshgrid(xa,ya)
C=np.sqrt((xv-lenth//2)**2+(yv-lenth//2)**2)
fig,axes=plt.subplots()
axes.scatter(xv,yv)
#axes.contour(xv,yv,C,cmap = plt.cm.hot)
#drew_cycle=plt.Circle((lenth//2 ,lenth//2), lenth//2, fill = False )
axes.set_aspect(1)
#axes.add_artist(drew_cycle)
axes.scatter(lenth//2 ,lenth//2,color='r')

plt.show()
#-----------------------
