import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

######### ortho

#Trained
#fname="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/ortho_and/trained/distribution_parameters.txt"
#fname="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/ortho_or/trained/distribution_parameters.txt"
#fname="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/ortho_xor/trained/distribution_parameters.txt"
#fname="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/ortho/ortho_ff/ff_trained/distribution_parameters.txt"

#Initial
#fname2="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/ortho_and/initial/distribution_parameters.txt"
#fname2="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/ortho_or/initial/distribution_parameters.txt"
#fname2="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/ortho_xor/initial/distribution_parameters.txt"
#fname2="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/ortho/ortho_ff/ff_initial/distribution_parameters.txt"

########### Normal

#Trained
#fname="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/normal_and/trained/distribution_parameters.txt"
#fname="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/normal/normal_xor/xor_trained/distribution_parameters.txt"
#fname="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/normal/normal_or/or_trained/distribution_parameters.txt"
fname="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/normal/normal_ff/trained_ff/distribution_parameters.txt"

#Initial
#fname2="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/normal_and/initial/distribution_parameters.txt"
#fname2="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/normal/normal_xor/xor_initial/distribution_parameters.txt"
#fname2="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/normal/normal_or/or_initial/distribution_parameters.txt"
fname2="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/paper_eigen_values/figures_paper/normal/normal_ff/initial_ff/distribution_parameters.txt"


#pepe    = np.loadtxt(fname,delimiter=' ')
pepe    =np.genfromtxt(fname,delimiter=' ')
#pepe    = np.genfromtxt(fname,delimiter="\t")
w       = pepe.T
print (pepe)

pepe2    =np.genfromtxt(fname2,delimiter=' ')
w2       = pepe2.T

########### Fig a) ############

regr = linear_model.LinearRegression()

regr.fit(np.array(w2[1]).reshape((-1, 1)),w[1])
y_pred = regr.predict(np.array(w2[1]).reshape((-1, 1)))
xData=w2[1]
yData=w[1]

absError = y_pred -w2[1]
SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(yData))
#a=regr.coef_[0]
b=regr.coef_[0]
b='%.4f'%b
RMSE='%.6f'%RMSE

fig     = plt.figure(figsize=cm2inch(12,6))

plt.plot([min(w2[1]),max(w2[1])],[min(w2[1]),max(w2[1])],'-',alpha=.15,color="grey",label="Trained = Initial")
plt.plot(w2[1],y_pred, color='blue',alpha=.15,linewidth=1,label="Linear regresion slope " +str(b)+"(+/-)"+str(RMSE))
#plt.plot([-0.002,0.004],[-0.002,0.004],'-',alpha=.15,color="grey",label="Trained = Initial")
plt.errorbar(w2[1],w[1], xerr=0, yerr=0,marker="o", markersize=2,fmt='o',color="salmon",label=r'$\mu$'+' Differences')

#plt.vlines(w2[1],w[1]-absError,w[1]+absError,alpha=.15,color="grey")

for i in range(len(xData)):
        lineXdata = (xData[i], xData[i]) # same X
        lineYdata = (yData[i], y_pred[i]) # different Y
        plt.plot(lineXdata, lineYdata,alpha=.15,color="grey")


#plt.axhline(y=1, color='pink', linestyle='--')
#plt.axhline(y=1.5, color='pink', linestyle='--')
plt.ylabel('Trained',fontsize = 10)
plt.xlabel('Initial',fontsize = 10)
plt.legend(fontsize=5,loc=2)
#plt.ylim([-0.25,8.1])

plt.xticks([])
plt.yticks([])
plt.xlim([min(w2[1])-0.01,max(w2[1])+0.01])
plt.ylim([min(w[1])-0.01,max(w[1])+0.01])
#plt.xticks(np.arange(0, max(lista_tot_porc__a)+2, 2.0),fontsize = 5)
#plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("figures_paper/distribution_par_mu.png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()

#########################################################


regr = linear_model.LinearRegression()
regr.fit(np.array(w2[2]).reshape((-1, 1)),w[2])
y_pred = regr.predict(np.array(w2[2]).reshape((-1, 1)))
xData=w2[2]
yData=w[2]

absError = y_pred -w2[2]
SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(yData))
#a=regr.coef_[0]
b=regr.coef_[0]
b='%.4f'%b
RMSE='%.6f'%RMSE

fig     = plt.figure(figsize=cm2inch(12,6))

#plt.plot([0.14135,0.14142],[0.14135,0.14142],'-',alpha=.15,color="grey",label="Trained = Initial")

plt.plot([min(w2[2]),max(w2[2])],[min(w2[2]),max(w2[2])],'-',alpha=.15,color="grey",label="Trained = Initial")
plt.plot(w2[2],y_pred, color='blue',alpha=.15,linewidth=1,label="Linear regresion slope " +str(b)+"(+/-)"+str(RMSE))
plt.errorbar(w2[2],w[2], xerr=0, yerr=0,marker="o", markersize=2,fmt='o',color="salmon",label=r'$\sigma$'+' Differences')

for i in range(len(xData)):
        lineXdata = (xData[i], xData[i]) # same X
        lineYdata = (yData[i], y_pred[i]) # different Y
        plt.plot(lineXdata, lineYdata,alpha=.15,color="grey")

plt.xticks([])
plt.yticks([])

#plt.axhline(y=1, color='pink', linestyle='--')
#plt.axhline(y=1.5, color='pink', linestyle='--')
plt.ylabel('Trained',fontsize = 10)
plt.xlabel('Initial',fontsize = 10)
plt.legend(fontsize=5,loc=2)
#plt.ylim([-0.25,8.1])
#plt.xlim([min(w2[2])-0.01,max(w2[2])+0.01])
#plt.ylim([min(w[2])-0.01,max(w[2])+0.01])

plt.xlim([min(w2[2])-0.01,max(w2[2])+0.01])
plt.ylim([min(w[2])-0.01,max(w[2])+0.01])

plt.xticks([])
plt.yticks([])
#plt.xticks(np.arange(0, max(lista_tot_porc__a)+2, 2.0),fontsize = 5)
#plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("figures_paper/distribution_sigma.png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()

######################################################################################

regr = linear_model.LinearRegression()
regr.fit(np.array(w2[3]).reshape((-1, 1)),w[3])
y_pred = regr.predict(np.array(w2[3]).reshape((-1, 1)))
xData=w2[3]
yData=w[3]

absError = y_pred -w2[3]
SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(yData))
#a=regr.coef_[0]
b=regr.coef_[0]
b='%.4f'%b
RMSE='%.6f'%RMSE

fig     = plt.figure(figsize=cm2inch(12,6))
#plt.plot([-1.34,-1.2],[-1.34,-1.2],'-',alpha=.15,color="grey",label="Trained = Initial")

plt.plot([min(w2[3]),max(w2[3])],[min(w2[3]),max(w2[3])],'-',alpha=.15,color="grey",label="Trained = Initial")
plt.plot(w2[3],y_pred, color='blue',alpha=.15,linewidth=1,label="Linear regresion slope " +str(b)+"(+/-)"+str(RMSE))
plt.errorbar(w2[3],w[3], xerr=0, yerr=0,marker="o", markersize=2,fmt='o',color="salmon",label=r'Kurtosis'+' Differences')

for i in range(len(xData)):
        lineXdata = (xData[i], xData[i]) # same X
        lineYdata = (yData[i], y_pred[i]) # different Y
        plt.plot(lineXdata, lineYdata,alpha=.15,color="grey")


#plt.axhline(y=1, color='pink', linestyle='--')
#plt.axhline(y=1.5, color='pink', linestyle='--')
plt.ylabel('Trained',fontsize = 10)
plt.xlabel('Initial',fontsize = 10)
plt.legend(fontsize=5,loc=2)

plt.xticks([])
plt.yticks([])
plt.xlim([min(w2[3])-0.01,max(w2[3])+0.01])
plt.ylim([min(w[3])-0.01,max(w[3])+0.01])
#plt.ylim([-0.25,8.1])
#plt.xlim([min(w2[1])-0.2*min(w2[1]),max(w2[1])+0.1*max(w2[1])])
#plt.ylim([min(w[1])-0.2*min(w[1]),max(w[1])+0.1*max(w[1])])
#plt.xticks(np.arange(0, max(lista_tot_porc__a)+2, 2.0),fontsize = 5)
#plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("figures_paper/distribution_kur.png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()

##################################################################################

regr = linear_model.LinearRegression()
regr.fit(np.array(w2[4]).reshape((-1, 1)),w[4])
y_pred = regr.predict(np.array(w2[4]).reshape((-1, 1)))
xData=w2[4]
yData=w[4]

absError = y_pred -w2[4]
SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(yData))
#a=regr.coef_[0]
b=regr.coef_[0]
b='%.4f'%b
RMSE='%.6f'%RMSE

fig     = plt.figure(figsize=cm2inch(12,6))

plt.plot([min(w2[4]),max(w2[4])],[min(w2[4]),max(w2[4])],'-',alpha=.15,color="grey",label="Trained = Initial")

plt.plot(w2[4],y_pred, color='blue',alpha=.15,linewidth=1, label="Linear regresion slope " +str(b)+"(+/-)"+str(RMSE))
plt.errorbar(w2[4],w[4], xerr=0, yerr=0,marker="o", markersize=2,fmt='o',color="salmon",label=r'Skew'+' Differences')

for i in range(len(xData)):
        lineXdata = (xData[i], xData[i]) # same X
        lineYdata = (yData[i], y_pred[i]) # different Y
        plt.plot(lineXdata, lineYdata,alpha=.15,color="grey")


plt.xticks([])
plt.yticks([])
#plt.axhline(y=1, color='pink', linestyle='--')
#plt.axhline(y=1.5, color='pink', linestyle='--')
plt.ylabel('Trained',fontsize = 10)
plt.xlabel('Initial',fontsize = 10)
plt.legend(fontsize=5,loc=2)
#plt.ylim([-0.25,8.1])
#plt.xlim([min(w2[1])-0.2*min(w2[1]),max(w2[1])+0.1*max(w2[1])])
#plt.ylim([min(w[1])-0.2*min(w[1]),max(w[1])+0.1*max(w[1])])
#plt.xticks(np.arange(0, max(lista_tot_porc__a)+2, 2.0),fontsize = 5)
#plt.yticks(np.arange(0,8.1,2),fontsize = 5)
plt.savefig("figures_paper/distribution_skew.png",dpi=300, bbox_inches = 'tight')
#plt.show()
plt.close()




