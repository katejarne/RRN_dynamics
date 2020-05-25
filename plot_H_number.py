import os
import time
import fnmatch

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.cm as cm


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


#r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/ortho"
r_dir="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/normal"

########### Fig a) ############
colors = cm.rainbow(np.linspace(0, 1, 10))

fig=plt.figure(figsize=cm2inch(7.1,6) )
ax = fig.add_axes([0, 0, 1, 1])  
cmap='viridis'

for root, sub, files in os.walk(r_dir):
    files = sorted(files)
    
    for i,f in enumerate(files):
        #print(f)
        if fnmatch.fnmatch(f, 'H_number_trained.txt'):#fnmatch.fnmatch(f, '*initial.hdf5') :#
           print("file: ",f)
           r_dir=root
           pepe    =np.genfromtxt(r_dir+"/"+f,delimiter=' ')
           print(pepe)
           w       = pepe.T
           histo_lista    =[]
           histo_lista.extend(w[1])
           print(histo_lista)
           media= np.average(histo_lista)
           media= "%.4f" % media
           #n, bins, patches = plt.hist(histo_lista, 10, normed=1, facecolor='green', alpha=0.75)
           plt.hist( histo_lista, alpha = 0.5,label="H Trained mean "+str(media))
           plt.legend(fontsize= 5,loc=1)

    for i,f in enumerate(files):
        #print(f)
        if fnmatch.fnmatch(f, 'H_number_ini.txt'):#fnmatch.fnmatch(f, '*initial.hdf5') :#
           print("file: ",f)
           r_dir=root
           pepe    =np.genfromtxt(r_dir+"/"+f,delimiter=' ')
           print(pepe)
           w       = pepe.T
           histo_lista    =[]
           histo_lista.extend(w[1])
           print(histo_lista)
           media= np.average(histo_lista)
           media= "%.4f" % media
           #n, bins, patches = plt.hist(histo_lista, 10, normed=1, facecolor='green', alpha=0.75)
           plt.hist( histo_lista, alpha = 0.5,label="H Ini  mean"+str(media))
           plt.legend(fontsize= 5,loc=1)

    
plt.xlabel('Henrici’s departure from normality',fontsize = 8)
#plt.axvline(x=0, linewidth=1, color='grey',alpha=.15)
plt.axvline(x=1, linewidth=1, color='grey',alpha=.15)
#plt.xticks([])
plt.yticks([])
#plt.xlim(-0.1,1.1)
#plt.xlim(-0.49,1.1)
#plt.ylim([min(w[1])-0.01,max(w[1])+0.01])
#plt.xticks(np.arange(0, 1.1,0.25),fontsize = 5)
plt.xticks(np.arange(4.9, 1.1,0.25),fontsize = 5)
#plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("figures_paper/H.png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()



