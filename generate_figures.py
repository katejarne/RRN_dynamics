import os
import time
import fnmatch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import grid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import grid
from scipy.stats import norm
from scipy.stats import norm, skew, kurtosis
from numpy import linalg as LA
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential,load_model
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import SimpleRNN
from keras.layers import TimeDistributed, Dense, Activation, Dropout
from keras.utils import plot_model, CustomObjectScope
from keras import metrics, optimizers, regularizers, initializers
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.constraints import Constraint

import tensorflow as tf

# taking dataset from function:

#from generate_data_set import *
from generate_data_set_and import *
#from generate_data_set_or import *
#from generate_data_set_xor import *
#from generate_data_set_not import *
#from generate_data_set_and_rand_place import *

# To print network status
from print_status_2_inputs_paper import *

#Parameters:
sample_size_3       = 4#15#4
mem_gap             = 20
sample_size         = 15 # Data set to print some results
lista_distancia_all =[]
lista_freq_sample_net=[]
H_number_list=[]
# Generate a data Set to study the Network properties:

x_train,y_train, mask,seq_dur  = generate_trials(sample_size,mem_gap) 
test                           = x_train[0:1,:,:] # Here you select from the generated data set which is used for test status
test_set                       = x_train[0:20,:,:]
y_test_set                     = y_train[0:20,:,0]
full_eigen_list                =[]
j2_full_eigen_list             =[]

dist_par_i                     =[]
dist_par_mu                    =[]
dist_par_sigma                 =[]
dist_par_pdf_kurtosis          =[]
dist_par_pdf_skew              =[]


net_freq=[]
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

#########################

#Path with network/s

#r_dir=""

plot_dir="figures_paper"

lista_neg     =[]
lista_pos     =[]
total         =[]
lista_neg_porc=[]
lista_pos_porc=[]
lista_tot_porc=[]

string_name_list=[]

for root, sub, files in os.walk(r_dir):
    files = sorted(files)
    
    for i,f in enumerate(files):
        print(f)
        if   fnmatch.fnmatch(f, '*05.hdf5'):# fnmatch.fnmatch(f, '*initial.hdf5') :#19
           print("file: ",f)
           r_dir=root
           string_name=root[-23:]
           print("string_name",string_name)
           string_name_list.append(string_name)
           print("r_dir",r_dir)

           #General network model construction:
           model = Sequential()
           model = load_model(r_dir+"/"+f)   #custom_objects={'NonNegLast':NonNegLast})
           # Compiling model for each file:
           model.compile(loss = 'mse', optimizer='Adam', sample_weight_mode="temporal")

           print("-------------",i)
           #Weights!!!
           for jj, layer in enumerate(model.layers):
               print("i-esima capa: ",jj)
               print(layer.get_config(), layer.get_weights())

           pesos     = model.layers[0].get_weights()
           pesos__   = model.layers[0].get_weights()[0]
           pesos_in  = pesos[0]
           pesos_out = model.layers[1].get_weights()
           pesos     = model.layers[0].get_weights()[1] 
           # biases   = model.layers[0].get_weights()[2]

           N_rec                          =len(pesos_in[0])  # it has to match the value of the recorded trained network
           neurons                        = N_rec
           colors                         = cm.rainbow(np.linspace(0, 1, neurons+1))


           print( "h",model.layers[0].states[0])

           print("-------------\n-------------")   
           print("pesos:\n:",pesos)
           print("-------------\n-------------")
           print("N_REC:",N_rec)
           unidades        = np.arange(len(pesos))
           conection       = pesos

           
           print("array: ",np.arange(len(pesos)))       
           #print("biases: ",biases)

           print("##########################\n ##########################")
           print("conection",conection)       
           print("##########################\n ##########################")

           histo_lista    =[]
           array_red_list =[]

           conection_usar =conection 
           conection_sym  =0.5*(conection+tf.transpose(conection))

           model.layers[0].set_weights([pesos_in,conection_usar])

           w, v = LA.eig(conection_usar)

           print("Eigenvalues:\n", w)
           print("Eigenvectors:\n",v)
           print("Distance:", np.sqrt(w.real*w.real+w.imag*w.imag))

           lista_dist  = np.c_[w,w.real]
           lista_dist_2= np.c_[w,abs(w.real)]
           maximo      = max(lista_dist, key=lambda item: item[1])

           maximo_2    = max(lista_dist_2, key=lambda item: item[1])
           marcar      = maximo[0]
           marcar_2    = maximo_2[0]

           print("First Element",maximo)
           print("Last Element",marcar)

           frecuency   =0
           if marcar_2.imag==0:
               frecuency =0
           else: 
               frecuency =abs(float(marcar_2.imag)/(3.14159*float(marcar_2.real)))

           print( "frecuency",frecuency)

           lista_modulos_    =np.sqrt(w.real*w.real+w.imag*w.imag)
           lista_freq_       =1000*np.absolute(w.imag/(3.14159*w.real))
           w_2               =list(w)

           list_dist_ordered =sorted(w_2, key=lambda x: abs(x.imag) )
           print("List sorted", list_dist_ordered)

           j2 = [i for i in w_2 if abs(i.real*i.real+i.imag*i.imag) > 1 and i.imag!=0]
          
           #print (j2)

           if len(j2)>0:
               ultimo= max(j2,key= np.abs)   #np.imag)
           else:
               ultimo =marcar_2
          # else:
          #    j2 = [i for i in w_2 if abs(i.real*i.real+i.imag*i.imag) >1]
          #    ultimo= max(j2,key= np.abs)

           #Debugging prints:
           #print("modulos",lista_modulos_)
           #print("j2 ",j2 )
           #print("j2 ultimo",ultimo )

           
           frecuency_ultimo =1000*abs(float(ultimo.imag)/(2*3.14159*float(ultimo.real)))
           net_freq.append([string_name,frecuency_ultimo])
           frecuency_ultimo_="%.2f" % frecuency_ultimo
           lista_modulos_cuad=  [i**2 for i in lista_modulos_]

           #Henrici’s departure from normality
           H_number=np.sqrt(np.power(np.linalg.norm(conection_usar),2)-sum(lista_modulos_cuad))/np.linalg.norm(conection_usar)
           H_number_list.append([string_name,H_number])

           #Symetric part

 
           w_s, v_s      = LA.eig(conection_sym)
           lista_dist_s  = np.c_[w_s,w_s.real]
           lista_dist_2_s= np.c_[w_s,abs(w_s.real)]
           maximo_s      = max(lista_dist_s, key=lambda item: item[1])

           maximo_2_s    = max(lista_dist_2_s, key=lambda item: item[1])
           marcar_s      = maximo_s[0]
           marcar_2_s    = maximo_2_s[0]

           ################ Fig Eigenvalues ########################

           fig=plt.figure(figsize=cm2inch(7.1,6) )
           ax = fig.add_axes([0, 0, 1, 1])         
           a = np.linspace(0, 2*np.pi, 500)
           cx,cy = np.cos(a), np.sin(a)
           plt.plot(cx, cy,'--', alpha=.25, color="dimgrey") # draw unit circle line
           plt.axvline(x=1,color="salmon",alpha=.25,linestyle='--')
           plt.plot([0,marcar.real],[0,marcar.imag],'-',alpha=.15,color="grey")
           plt.plot([0,ultimo.real],[0,ultimo.imag],'-',alpha=.15,color="grey")

           t=w.real
           plt.scatter(w.real,w.imag,c=t,cmap='viridis',s=2)
           plt.scatter(ultimo.real,ultimo.imag,color="blue",label="Max Eigenvalue Comp.\n "+str(ultimo),s=5)
           plt.scatter(marcar.real,marcar.imag,color="red", label="Max Eigenvalue Real \n" +str(marcar_2)+"\n"+" Freq: "+str(frecuency_ultimo_),s=5)
                  
           plt.xticks(fontsize=6)
           plt.yticks(fontsize=6)
           plt.ylim([-1.3, 1.6])
           plt.xlim([-1.5, 1.6])
           plt.xlabel(r'$Re( \lambda)$',fontsize = 11)
           plt.ylabel(r'$Im( \lambda)$',fontsize = 11) 

           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           ax.spines['bottom'].set_visible(False)
           ax.spines['left'].set_visible(False)
           ax.get_xaxis().set_ticks([])
           ax.get_yaxis().set_ticks([])
           
           leg = plt.legend(fontsize= 5,loc=1)
           leg.get_frame().set_linewidth(0.0)
           #leg = plt.legend()
           #leg.get_frame().set_linewidth(0.0)    
           plt.savefig(plot_dir+"/autoval_"+str(i)+"_"+str(f)+"_"+str(string_name)+".png",dpi=300, bbox_inches = 'tight')
           plt.close()
  
           #########################################################
           
           fig=plt.figure(figsize=cm2inch(7.1,6))
           ax = fig.add_axes([0, 0, 1, 1])     
           t_=w_s.real
           plt.scatter(w_s.real,w_s.imag,c=t_,cmap='viridis',label="Eigenvalue Sym spectrum\n Max: "+str(marcar_2_s),s=2)
           a = np.linspace(0, 2*np.pi, 500)
           cx,cy = np.cos(a), np.sin(a)
           plt.plot(cx, cy,'--', alpha=.25, color="dimgrey")           
           plt.scatter(marcar_s.real,marcar_s.imag,color="red", label="Eigenvalue maximum real part",s=5)
           plt.plot([0,marcar_s.real],[0,marcar_s.imag],'-',color="grey",alpha=.15)           
           plt.axvline(x=1,color="salmon",alpha=.25,linestyle='--')
           plt.xticks(fontsize=6)
           plt.yticks(fontsize=6)
           plt.ylim([-1.3, 1.6])
           plt.xlim([-1.5, 1.6])
           plt.xlabel(r'$Re( \lambda)$',fontsize = 11)
           plt.ylabel(r'$Im( \lambda)$',fontsize = 11)
           #plt.legend(fontsize= 8,loc=1)            
           leg = plt.legend(fontsize= 5,loc=1)
           leg.get_frame().set_linewidth(0.0)
           #leg = plt.legend()
           #leg.get_frame().set_linewidth(0.0)
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           ax.spines['bottom'].set_visible(False)
           ax.spines['left'].set_visible(False)
           ax.get_xaxis().set_ticks([])
           ax.get_yaxis().set_ticks([])
    
           plt.savefig(plot_dir+"/autoval_sym_"+str(i)+"_"+str(f)+"_"+str(string_name)+".png",dpi=300, bbox_inches = 'tight')
           plt.close()
   
           #################################################################
           # Histogram estimations:

           fig=plt.figure(figsize=cm2inch(7,5.5))
           ax = fig.add_axes([0, 0, 1, 1])     
           for ii in unidades:
               histo_lista.extend(pesos[ii])

           media= np.average(histo_lista)


           # best fit of data
           (mu, sigma) = norm.fit(histo_lista)
           # the histogram of the data

           n, bins, patches = plt.hist(histo_lista, 200, normed=1, facecolor='green', alpha=0.75)
           
           y =  norm.pdf( bins, mu, sigma)
           
           pdf_kurtosis = kurtosis(y)
           pdf_skew     = skew(y)

           dist_par_i.append(string_name)
           dist_par_mu.append(mu)
           dist_par_sigma.append(sigma)
           dist_par_pdf_kurtosis.append(pdf_kurtosis)
           dist_par_pdf_skew.append(pdf_skew)

           mu_="%.4f" % mu
           sigma_= "%.4f" % sigma
           pdf_skew_= "%.4f" % pdf_skew
           pdf_kurtosis_="%.4f" % pdf_kurtosis
                     
           #if i==0:
           #    plt.title('Initial Histogram Weights', fontsize = 18)
           #else:
           #plt.title('Histogram Weights after \"AND\" learning', fontsize = 12)
           #plt.hist(histo_lista, bins=200,color="mistyrose",normed=1,label="Weight Value \n Mu= "+str(mu)+"\n Sigma= "+str(sigma))
           plt.axvline(mu, color='r', linestyle='dashed', linewidth=1, label=r'$W^{Rec}$ Values'+'\n\n'+ r'$\mu$= '+str(mu_)+'\n $\sigma= $ '+str(sigma_)+"\n Skew= " +str(pdf_skew_)+"\n Kurtosis= "+str(pdf_kurtosis_))

           plt.plot(bins, y, 'r-', linewidth=1)
           plt.vlines(x=sigma, ymin=0, ymax=900, linewidth=1, color='grey',alpha=.15)
           plt.vlines(x=-sigma,ymin=0, ymax=900, linewidth=1, color='grey',alpha=.15)
           
           n = n.astype('int') # it MUST be integer
           # Good old loop. Choose colormap of your taste
           for i_pa in range(len(patches)):
               patches[i_pa].set_facecolor(plt.cm.viridis(n[i_pa]/max(n)))

           plt.xlabel('Weight strength [arb. units]',fontsize = 8)
           plt.ylim([0,4.5])
           plt.xlim([-0.5,0.5])
           #plt.legend(fontsize= 8,loc=1)
           leg = plt.legend(fontsize= 5,loc=1)
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           ax.spines['bottom'].set_visible(False)
           ax.spines['left'].set_visible(False)
           ax.get_xaxis().set_ticks([])
           ax.get_yaxis().set_ticks([])

           leg.get_frame().set_linewidth(0.0)
           plt.savefig(plot_dir+"/weight_histo_"+str(string_name)+"_"+str(f)+"_.png",dpi=300, bbox_inches = 'tight')
           plt.close()

           ################################################################################
           fig=plt.figure(figsize=cm2inch(7.5,5.5))
           #plt.title('Histogram Weights in after \"AND\" learning', fontsize = 11)
           ax = fig.add_axes([0, 0, 1, 1])     
           plt.hist(pesos_in[0], bins=200,label="Weight")
           plt.xlabel('Weight strength [arb. units]',fontsize = 8)
           leg = plt.legend(fontsize= 6,loc=1)
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           ax.spines['bottom'].set_visible(False)
           ax.spines['left'].set_visible(False)
           leg.get_frame().set_linewidth(0.0)
           plt.savefig(plot_dir+"/in_weight_histo_"+str(i)+"_"+str(f)+"_.png",dpi=300, bbox_inches = 'tight')
           plt.close()

           ##############################################################################     

                   

           ########## Here we plot iner state of the network with the desierd stimuli: 
           
           for sample_number in np.arange(sample_size_3):
           #   #if sample_number==2:#2#3xor
              print ("sample_number",sample_number)
              print_sample = plot_sample(sample_number,2,neurons,x_train,y_train,model,seq_dur,i,plot_dir,f,string_name)
              lista_freq_sample_net.append([string_name,sample_number,print_sample])
           #print("print_sample",print_sample)
           #time.sleep(5)
           ########## 
           

           # Model Testing: 
           x_pred = x_train[0:10,:,:]
           y_pred = model.predict(x_pred)

           print("x_train shape:\n",x_train.shape)
           print("x_pred shape\n",x_pred.shape)
           print( "y_train shape\n",y_train.shape)

           lista_distancia=[]
           #################################################

           for ii in np.arange(10):
               a=y_train[ii, :, 0]
               b=y_pred[ii, :, 0]          
               a_min_b = np.linalg.norm(a-b)      
               lista_distancia.append(a_min_b)

       
           lista_distancia.insert(0,N_rec)
           lista_distancia_all.append(lista_distancia)       

           #########################################
           fig= plt.figure(figsize=cm2inch(9.5,9))
           #fig.suptitle("\"And\" Data Set Trainined Output \n (amplitude in arb. units time in mSec)",fontsize = 20)
           for ii in np.arange(6):

               a=y_train[ii, :, 0]
               b=y_pred[ii, :, 0]
               a_min_b = np.linalg.norm(a-b)  
               a_min_b_="%.4f" %a_min_b 
               #lista_distancia.append(a_min_b)
               plt.subplot(3, 2, ii + 1)                
               plt.plot(x_train[ii, :, 0],color='g',label="Input A")
               plt.plot(x_train[ii, :, 1],color='pink',label="Input B")
               plt.plot(y_train[ii, :, 0],color='gray',linewidth=2,label="Expected Output")
               plt.plot(y_pred[ii, :, 0], color='r',linewidth=1,label="Predicted Output\n Distance= "+str(a_min_b_))
               #plt.ylim([-2, 1.6])
               plt.ylim([-2.5, 2])
               #plt.xlim([0, 205])
               #plt.xlim([0, 205])
               plt.xticks(np.arange(0,205,100),fontsize = 8)
               #plt.xticks(np.arange(0,405,100),fontsize = 8)
               #plt.legend(fontsize= 4.75,loc=3)
               leg = plt.legend(fontsize= 3.5,loc=3)
               leg.get_frame().set_linewidth(0.0)
               #plt.xticks([])
               plt.yticks([])
               plt.xticks(fontsize=5)
               plt.yticks(fontsize=5)
           fig.text(0.5, 0.03, 'time [mS]',fontsize=5, ha='center')
           fig.text(0.1, 0.5, 'Amplitude [Arb. Units]', va='center', ha='center', rotation='vertical', fontsize=5)
           figname =plot_dir+"/data_set_"+str(f)+"_"+str(string_name)+"_.png"       
           plt.savefig(figname,dpi=300, bbox_inches = 'tight')

           K.clear_session()

todo_2       = np.c_[lista_distancia_all]
np.savetxt(plot_dir+'/distance_sample.txt',todo_2,fmt='%f %f %f %f %f %f %f %f %f %f %f',delimiter=' ',header="Nrec #S1 #S2 #S3 #S4 #S5 #S6 #S7 #S8 #S9 #S10")
np.savetxt(plot_dir+'/freq_net.txt',net_freq,fmt='%s %s',delimiter='\t',header="Net-id #freq ")
np.savetxt(plot_dir+'/freq_net_units.txt',lista_freq_sample_net,fmt='%s',delimiter='\t',header="Net-id #sample #freq ")

todo=np.c_[dist_par_i, dist_par_mu, dist_par_sigma, dist_par_pdf_kurtosis, dist_par_pdf_skew]
#print("todo",todo)
np.savetxt(plot_dir+'/distribution_parameters.txt' ,todo,fmt='%s %s %s %s %s',delimiter=' ',header='Net-id, #mu #sigma #kurtosis #skew')
np.savetxt(plot_dir+'/H_number.txt',H_number_list,fmt='%s %s',delimiter='\t',header="Name #H number ")

print ("distancias",todo_2)
print("freq de las redes",net_freq)
print("freq unidades",lista_freq_sample_net)

