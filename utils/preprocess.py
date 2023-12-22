import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#get static attribute table in which values are not repeated
def get_static(data,columns,subject_idx='subject_id'):
    return data.groupby(subject_idx)[columns].first()

#get 2d timevarying data to 3d numpy array
def df_to_3d(data,columns,timestep_idx='seq_num',subject_idx='subject_id'):
    t = data[timestep_idx].max()+1
    n = data[subject_idx].nunique()
    data_3d = np.full(shape=(n,t,len(columns)),fill_value=-1)
    for idx,(_,subject) in enumerate(data.groupby(subject_idx)[columns]):
        #subject data has shape (t,k)
        data_3d[idx,:subject.shape[0],:] = subject
    return data_3d

#one hot encode non-binary categoricals
def one_hot_encoding(data,columns):
    for i in columns:
        dummies = pd.get_dummies(data[i],prefix=str(i))
        data = data.drop(i,axis=1)
        data = pd.concat([data,dummies],axis=1)
    return data

#normalizing function for pandas dataframe
def normalize(x):
    return (x-x.mean())/(x.std())

#zero one scaling function for pandas dataframe
def zero_one_scale(x):
    return (x-x.min())/(x.max()-x.min())


def train_split(X,y,stratify=None,train_size=.7):
    #create train test split stratified on syn/real labels
    return train_test_split(X,y,stratify=stratify,train_size=train_size)