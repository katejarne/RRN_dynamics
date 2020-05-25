#A code for print network status when different data set input samples are applied
# Plot of Individual neural state for the interation that you defined in load and print
# Plot of SVD in 2 and 3D
# Plot of PCA in 3D

from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
from pylab import grid
from matplotlib.cbook import get_sample_data
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import grid
from keras import backend as K
from keras.models import Sequential, Model
from scipy.stats import norm
from scipy import signal

# pca part:
from sklearn.decomposition import PCA
import sklearn.decomposition
from scipy import signal

#/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/
#im = Image.open('/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figs_gates/and.png')
#im = Image.open('/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figs_gates/xor.png')
#im = Image.open('/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figs_gates/or.png')
im = Image.open('figs_gates/ff_.png')
#height = im.size[1]*0.75
#if ff
height = im.size[1]*0.24
# We need a float array between 0-1, rather than
# a uint8 array between 0-255
im = np.array(im).astype(np.float) / 255



def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_sample(sample_number,input_number,neurons,x_train,y_train,model,seq_dur,i,plot_dir,f,string_name):

    frecuencias=[]
    
    seq_dur                        = len(x_train[sample_number, :, 0])
    test                           = x_train[sample_number:sample_number+1,:,:]
    cmap = plt.get_cmap('viridis')
    colors                         = cm.rainbow(np.linspace(0, 1, neurons+1))
    y_pred                         = model.predict(test)
    

    ###################################

    # Status for the sample value at the layer indicated
    capa=0

    #Primer capa:
    get_0_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
    
    layer_output= get_0_layer_output([test])[capa]
        
    #segunda capa:
    get_1_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
    #layer_output_1     = get_1_layer_output([test])[capa]
                
    #tercer capa
    #get_2_layer_output = K.function([model.layers[0].input], [model.layers[2].output])
    #layer_output_2     = get_2_layer_output([test])[0]
    
    layer_output_T       = layer_output.T

    print("layer_output",layer_output_T)
    #layer_output_1_T     = layer_output_1.T
    #layer_output_2_T    = layer_output_2.T
    array_red_list       = []

    ####################################
    y_pred              = model.predict(test)

    # To generate the Populational Analysis

    for ii in np.arange(0,neurons,1):
        neurona_serie = np.reshape(layer_output_T[ii], len(layer_output_T[ii]))
        array_red_list.append(neurona_serie)
    
    array_red = np.asarray(array_red_list)
    sdv       = sklearn.decomposition.TruncatedSVD(n_components=2)
    sdv_3d    = sklearn.decomposition.TruncatedSVD(n_components=3)
    X_2d      = sdv.fit_transform(array_red.T)
    X_3d      = sdv_3d.fit_transform(array_red.T)
    pca       = PCA(n_components=3)
    X_pca_    = pca.fit(array_red)
    X_pca     = pca.components_

    ####################################

    print("------------")
    ordeno_primero_x=X_pca[0]
    ordeno_primero_y=X_pca[1]
    ordeno_primero_z=X_pca[2]
              
    ####################################
    # How many 3d angular views you want to define
    yy        = np.arange(70,80,10)

    ####################################   
    
    kk=70       

    fig =plt.figure(figsize=cm2inch(19,8))
   
    ###
    plt.subplot(4, 2, 1) #1
    #ax = fig.add_axes([0, 0, 1, 1])
    plt.plot(test[0,:,0],color='g',label='Input A',linewidth=1)
    plt.xlim(0,seq_dur+1)
    plt.ylim([-1.4, 1.2])
    plt.yticks([])
    plt.xticks(np.arange(0,seq_dur+1,50),fontsize = 8)
    plt.legend(fontsize= 5,loc=1)   
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    ##ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])
    plt.axis('off')
    plt.box(on=None)


    plt.subplot(4, 2, 3) #2
    plt.plot(test[0,:,1],color='pink',label='Input B',linewidth=1)
    plt.yticks([])
    plt.xticks(np.arange(0,seq_dur+1,50),fontsize = 8)
    plt.xlim(0,seq_dur+1)
    plt.ylim([-1.4, 1.2])
    plt.axis('off')
    plt.box(on=None)
    plt.legend(fontsize= 5,loc=1)

    plt.subplot(4, 2, 5)#3   
    plt.xlim(0,seq_dur+1)
    plt.ylim([-1.4, 1.1])
    #plt.plot(y_train[sample_number,:, 0],color='grey',linewidth=3,label='Target Output',alpha=.25)  
    plt.plot(y_pred[0,:, 0], color='r',label=' Output',linewidth=1) 
    plt.yticks([])
    plt.xticks(np.arange(0,seq_dur+1,50),fontsize = 8)
    plt.legend(fontsize= 5,loc='upper right',bbox_to_anchor=(1.14,1.5))
    plt.axis('off')
    plt.box(on=None)
    ### 
    plt.subplot(4, 2, 7) #4
    
    #plt.plot(y_train[sample_number,:, 0],color='grey',linewidth=3,label='Target Output')  
    plt.xlim(0,seq_dur+1)
    plt.ylim([-1.1, 1.1])   
    for ii in np.arange(0,int(neurons),1):
    
        plt.plot(layer_output_T[ii],color=colors[ii],linewidth=1)
        
        plt.xlabel('time [mS]',fontsize = 10)
        plt.yticks([])
        plt.xticks(np.arange(0,seq_dur+1,50),fontsize = 8)
    plt.plot(y_pred[0,:, 0], color='r',linewidth=1,label=' Output\n 25 individual states')    
    #plt.legend(fontsize= 3.5,loc=3)
    leg = plt.legend(fontsize= 5,loc='upper right',bbox_to_anchor=(1.14,1.5))
    #leg.get_frame().set_linewidth(0.0)
    #plt.axis('off')
    plt.box(on=None)

           
    #plt.subplot(2, 2, 4) 
    #fig.suptitle("Time series and PCA 3D plot",fontsize = 10)
    ax = fig.add_subplot(122, projection='3d')
    #fig.figimage(im,1300,20 ,fig.bbox.ymax - height)


    x=X_pca[0]
    y=X_pca[1]
    z=X_pca[2]
    N=len(z)
    
    
    if sample_number==1:
       ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=70,c='r',marker="^",label=' Start ')     
       ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=70,c='b',marker="^",label=' Stop ')

    if sample_number !=1:
       #ax.plot(X_pca[0],X_pca[1],X_pca[2],color='salmon',marker="p",zorder=2,markersize=2,label="3d plot")
       for ik in range(N-1):
        ax.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2], color=plt.cm.viridis(ik/N))
       ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=70,c='r',marker="^",label=' Start ')     
       ax.scatter(ordeno_primero_x[-1],ordeno_primero_y[-1],ordeno_primero_z[-1],s=70,c='b',marker="^",label=' Stop ')
      
    
    #ax.plot(X_pca[0],X_pca[1],X_pca[2],color='salmon',marker="p",zorder=2,markersize=2,label="3d plot")
   
    #ax.set_xlabel('pca 1 [arb. units]',size=10)
    #ax.set_ylabel('pca 2 [arb. units]',size=10)
    #ax.set_zlabel('pca 3 [arb. units]',size=10)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_zticks(())
    ax.view_init(elev=10, azim=kk)

    ax.legend(fontsize= 6)
    fig.text(0.1, 0.5, 'Amplitude [Arb. Units]', va='center', ha='center', rotation='vertical', fontsize=10)
    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_n_states_"+str(i)+"_satge_"+str(f)+"_"+str(string_name)+".png"
    plt.savefig(figname,dpi=300, bbox_inches = 'tight') 
    plt.close()      
    ####################################

    for ii in np.arange(0,int(neurons),1):
        a_ver=layer_output_T[ii]
        #a_ver=a_ver[100:]
        t_= np.arange(len(a_ver))
        freq=0
        print ("print t",t_)
    
        peakind_min,peakind_min_2      = signal.argrelmin(a_ver,axis=0)
        amp_pico_min_x   = a_ver[peakind_min]

        print( "peakind_min: ",peakind_min)
        print( "amp_pico_min_x: ",amp_pico_min_x)

        if len(amp_pico_min_x)>1:
            pepe         =t_[peakind_min]
            print ("pepe",pepe)
            if len(pepe)>6:
                freq         =1/(0.001*float(pepe[-2]-pepe[-3]))
        else:
            freq =0   

        ff='%.2f'%freq
        print("t",t_)
        print("Frequency",ff)
        frecuencias.append(freq)
    ######################################################################

    fig     = plt.figure(figsize=cm2inch(15,7))

    for ii in np.arange(0,int(4),1):
        plt.plot(layer_output_T[ii],color=colors[7*ii],linewidth=1,label="freq= "+str(frecuencias[ii]))
        plt.scatter(peakind_min,amp_pico_min_x,c="k")
        if len(peakind_min)>3:
            plt.scatter(peakind_min[-3],amp_pico_min_x[-3],c="b",marker="^")
            plt.scatter(peakind_min[-2],amp_pico_min_x[-2],c="b",marker="^")
            plt.vlines(x=peakind_min[-3],ymin=amp_pico_min_x[-3], ymax=0,color="grey",linestyle='--')
            plt.vlines(x=peakind_min[-2],ymin=amp_pico_min_x[-2], ymax=0,color="grey",linestyle='--')
    plt.xlim(-1,seq_dur+1)         
    plt.ylim([-1.5, 1.5])
    plt.xlabel('time [mS]',fontsize = 10)
    plt.ylabel('amplitude [arb. units]',fontsize = 10)
    plt.yticks([])
    plt.xticks(np.arange(0,seq_dur+1,50),fontsize = 8)

    plt.plot(y_pred[0,:, 0], color='r',linewidth=2,label=' Output\n 25 individual states')    
    #plt.legend(fontsize= 3.5,loc=3)
    leg = plt.legend(fontsize= 5,loc=3)
    #leg.get_frame().set_linewidth(0.0)

    figname = str(plot_dir)+"/S_"+str(sample_number)+"_n_states_"+"_stage"+str(f)+"_"+str(string_name)+".png"
    #plt.savefig(figname,dpi=300, bbox_inches = 'tight') 
    plt.close()   


    ######################################
    if len(frecuencias)>4:
        print("f",frecuencias) 
        #media= np.average(frecuencias)


    fig=plt.figure(figsize=cm2inch(7,5.5))
    plt.title('Histogram freq for \"AND\" learned units', fontsize =8)
    ax = fig.add_axes([0, 0, 1, 1])         
    n, bins, patches =plt.hist(frecuencias, bins=20,color="cyan",label="frequencies")#,label="Weight Value \n Mu= "+str(mu)+"\n Sigma= "+str(sigma))


    n = n.astype('int') # it MUST be integer
    # Good old loop. Choose colormap of your taste
    for i_pa in range(len(patches)):
        patches[i_pa].set_facecolor(plt.cm.viridis(n[i_pa]/max(n)))

    plt.xlabel('freq  [1/s]',fontsize = 16)

    leg = plt.legend(fontsize= 6,loc=1)
    leg.get_frame().set_linewidth(0.0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.savefig(plot_dir+"/freq_histo_"+str(sample_number)+"_"+str(f)+'_'+str(string_name)+"_.png",dpi=300, bbox_inches = 'tight')
    plt.close()

    ####################################################################################################################
      
    return frecuencias










