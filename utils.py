import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from dtwParallel import dtw_functions
from gower import gower_matrix

#get descriptive statistics
def descr_stats(data):
    stats = ['mean','std','min','max']
    list_ = []
    list_.append(data.mean(axis=0))
    list_.append(data.std(axis=0))
    list_.append(data.min(axis=0))
    list_.append(data.max(axis=0))
    return pd.DataFrame(list_,index=stats)

#get static attribute table in which values are not repeated
def get_static(data,columns,subject_idx='subject_id'):
    return data.groupby(subject_idx)[columns].first()

#get relative frequencies of categorical variables
def relative_freq(data):
    return pd.concat([data[col].value_counts(normalize=True) for col in data],axis=1)

#get matrixplot of relative frequencies of categorical variables over time
def rel_freq_matrix_plot(data,columns,timestep_idx='seq_num'):
    rel_freq = data.groupby(timestep_idx)[columns].value_counts(normalize=True).rename('rel_freq').reset_index()
    rel_freq = rel_freq.pivot(index=columns,columns=timestep_idx,values='rel_freq')
    plt.figure(figsize=(10, 6))
    sns.heatmap(rel_freq, annot=False, cmap='rocket_r', fmt=".2f")
    plt.title('Relative Frequency of Categories over Timesteps')
    plt.xlabel('Timestep')
    plt.ylabel('Category')
    return plt

#get 2d timevarying data to 3d numpy array
def df_to_3d(data,timevarying_cols,timestep_idx='seq_num',subject_idx='subject_id'):
    t = data[timestep_idx].max() 
    n = data[subject_idx].nunique()
    data_3d = np.ones((n,t,len(timevarying_cols)))*-1
    for idx,(_,subject) in enumerate(data.groupby(subject_idx)[timevarying_cols]):
        data_3d[idx,:subject.shape[0],:] = subject
    return data_3d

#get gower matrix for 2d data
def gower_matrix(data):
    return gower_matrix(data)

#get gower matrix for 3d time series of mixed datatypes
def mts_gower_matrix(data):
    class Input:
        def __init__(self):
            self.check_errors = False 
            self.type_dtw = "d"
            self.constrained_path_search = 'itakura'
            self.MTS = True
            self.regular_flag = -1
            self.n_threads = -1
            self.local_dissimilarity = "gower"
            self.visualization = False
            self.output_file = False
            self.dtw_to_kernel = False
            self.sigma_kernel = 1
            self.itakura_max_slope = None
            self.sakoe_chiba_radius = 1
    input_obj = Input()
    #to see progress, we can import tqdm and use it in dtwParallel package -> @ dtw_functions.dtw_tensor_3d 
    timevarying_distance = dtw_functions.dtw_tensor_3d(data,data,input_obj)
    return timevarying_distance

#one hot encode non-binary categoricals
def one_hot_encoding(data,columns):
    for i in columns:
        dummies = pd.get_dummies(data[i],prefix=str(i))
        data = data.drop(i,axis=1)
        data = pd.concat([data,dummies],axis=1)
    return data