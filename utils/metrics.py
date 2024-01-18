from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,accuracy_score,roc_auc_score
from matplotlib import pyplot as plt 
import seaborn as sns
import numpy as np 
import pandas as pd
from scipy.stats import ks_2samp

def mape(true,pred):
    return mean_absolute_percentage_error(true,pred)

def mae(true,pred):
    return mean_absolute_error(true,pred)

def accuracy(true,pred):
    return accuracy_score(true,pred)

def auc(labels,pred_scores):
    return roc_auc_score(labels,pred_scores)

def ks_test(real_pred,syn_pred):
    return ks_2samp(data1=real_pred.flatten(),data2=syn_pred.flatten(),alternative='two-sided')

#get descriptive statistics
def descr_stats(data):
    stats = ['mean','std','min','max']
    list_ = []
    list_.append(data.mean(axis=0))
    list_.append(data.std(axis=0))
    list_.append(data.min(axis=0))
    list_.append(data.max(axis=0))
    return pd.DataFrame(list_,index=stats,columns=data.columns)


#get relative frequencies of categorical variables
def relative_freq(data):
    proportions = pd.concat([data[col].value_counts(normalize=True) for col in data],axis=1)
    proportions.columns = data.columns
    return proportions

#get matrixplot of relative frequencies of categorical variables over time
def rel_freq_matrix(data,columns,timestep_idx='seq_num'):
    rel_freq = data.groupby(timestep_idx)[columns].value_counts(normalize=True).rename('rel_freq').reset_index()
    rel_freq = rel_freq.pivot(index=columns,columns=timestep_idx,values='rel_freq')
    return rel_freq

def freq_matrix_plot(rel_freq,range=None):
    if range != None:
        vmin,vmax = range
    else:
        vmin = vmax = None
    plt.figure(figsize=(10, 6))
    sns.heatmap(rel_freq, annot=False, cmap='rocket_r', fmt=".2f", vmin=vmin, vmax=vmax)
    plt.xlabel('Timestep')
    plt.ylabel('Category')
    #plt.xticks(ticks=np.arange(1,data[timestep_idx].max()+1,1),labels=data[timestep_idx].unique())
    #plt.yticks(ticks=np.arange(0,data[columns].max(),1),labels=np.sort(data[columns].unique()))
    return plt

def GoF_kdeplot(pred,y_test):
    #df = pd.DataFrame(np.concatenate((pred,np.expand_dims(y_test,-1)),axis=1),columns=['pred','hue'])
    #df = pd.DataFrame(np.concatenate((pred[y_test==0],pred[y_test==1]),axis=1),columns=['Real','Synthetic'])
    plt.hist(pred[y_test==0],bins='auto',label='Real',color='b',alpha=.5)
    plt.hist(pred[y_test==1],bins='auto',label='Synthetic',color='r',alpha=.5)
    plt.legend()
    return plt 