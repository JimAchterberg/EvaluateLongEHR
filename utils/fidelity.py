import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from gower import gower_matrix
from dtwParallel import dtw_functions
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score

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

def freq_matrix_plot(rel_freq,range=(0,0.2)):
    vmin,vmax = range
    plt.figure(figsize=(10, 6))
    sns.heatmap(rel_freq, annot=False, cmap='rocket_r', fmt=".2f", vmin=vmin, vmax=vmax)
    plt.title('Relative Frequency of Categories over Timesteps')
    plt.xlabel('Timestep')
    plt.ylabel('Category')
    #plt.xticks(ticks=np.arange(1,data[timestep_idx].max()+1,1),labels=data[timestep_idx].unique())
    #plt.yticks(ticks=np.arange(0,data[columns].max(),1),labels=np.sort(data[columns].unique()))
    return plt

#get gower matrix for 2d data
def static_gower_matrix(data,cat_features=None):
    return gower_matrix(data,cat_features=cat_features)

#get gower matrix for 3d time series of mixed datatypes
def mts_gower_matrix(data,cat_features=None):
    class Input:
        def __init__(self):
            self.check_errors = True 
            self.type_dtw = "d"
            self.constrained_path_search = None
            self.MTS = True
            self.regular_flag = '-1'
            self.n_threads = -1
            self.local_dissimilarity = "gower"
            self.visualization = False
            self.output_file = False
            self.dtw_to_kernel = False
            self.sigma_kernel = 1
            self.itakura_max_slope = None
            self.sakoe_chiba_radius = None
    input_obj = Input()
    #to see progress, we can import tqdm and use it in dtwParallel package -> @ dtw_functions.dtw_tensor_3d 
    timevarying_distance = dtw_functions.dtw_tensor_3d(data,data,input_obj,cat_features)
    return timevarying_distance


#tSNE projection plot, color coded by label
def tsne(distance_matrix,labels):
    embeddings = TSNE(n_components=2,init='random',metric='precomputed').fit_transform(distance_matrix)
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings[:,0],embeddings[:,1],c=labels)
    plt.title('tSNE plot')
    return plt

import keras
from keras import layers
class gof_model(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(100,activation='relu')
        self.recurrent = layers.LSTM(100,activation='relu')
        self.concat = layers.Concatenate(axis=1)
        self.process_1 = layers.Dense(100,activation='relu')
        self.process_2 = layers.Dense(50,activation='relu')
        self.classify = layers.Dense(1,activation='sigmoid')

    def call(self, inputs):
        attributes,longitudinal = inputs
        attr = self.dense(attributes)
        long = self.recurrent(longitudinal)
        x = self.concat([attr,long])
        x = self.process_1(x)
        x = self.process_2(x)
        return self.classify(x)

def gof_test(real_pred,syn_pred):
    return ks_2samp(data1=real_pred.flatten(),data2=syn_pred.flatten(),alternative='two-sided')

def accuracy(real,pred):
    return accuracy_score(real,pred)