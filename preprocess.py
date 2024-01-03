#general preprocessing necessary for most models:
#1: one hot encode race and icd codes
#2: separate static dataframe and 3d numpy array (padded to maximum timesteps)
#3: create train test split
#4: scale numerical variables (age)
#save this data to disk for easy use 
#models only need to create labels from train/test data and build/assess models

import pandas as pd
import os
import numpy as np
from utils import preprocess
import pickle

def general_preprocess(real_df,syn_df,save_path=None):
    #ONE HOT ENCODE
    df = pd.concat([real_df,syn_df],axis=0)
    for col in ['race','icd_code']:
        dummies = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,dummies],axis=1)
        df = df.drop(col,axis=1)
    n_real = real_df.subject_id.nunique()
    n_syn = syn_df.subject_id.nunique()
    #------------------------------------------------------------------------------ 
    #SEPARATE SEQUENTIAL/STATIC
    static_columns = [x for x in df.columns if 'race' in x or x in ['age','gender','deceased']]
    dynamic_columns = [x for x in df.columns if 'icd_code' in x]
    static = preprocess.get_static(df,static_columns)
    seq = preprocess.df_to_3d(df,dynamic_columns,padding=0)
    assert static.shape[0] == seq.shape[0]
    X_real = [static[:n_real],seq[:n_real]]
    X_syn = [static[n_real:],seq[n_real:]]
    #------------------------------------------------------------------------------ 
    #CREATE TRAIN TEST SPLIT
    #separately for real/synthetic, stratify on mortality,race,gender
    #strat_columns = [x for x in df.columns if 'race' in x or x in ['deceased','gender']]
    strat_columns = ['deceased','gender']
    X = []
    for data in [X_real,X_syn]:
        train_idx,test_idx,y_train,y_test = preprocess.train_split(
            np.arange(data[0].shape[0]),data[0].deceased,stratify=data[0][strat_columns],train_size=.7)
        
        for idx in [train_idx,test_idx]:
            x_static = data[0].iloc[idx].copy()
            x_seq = data[1][idx]
            x_static['age'] = preprocess.zero_one_scale(x_static['age'])
            X.append([x_static,x_seq])
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = X[0],X[1],X[2],X[3]
   
    #------------------------------------------------------------------------------ 
    #SAVE DATA TO DISK
    if save_path != None:
        for data,name in zip([X_real_tr,X_real_te,X_syn_tr,X_syn_te],['X_real_tr','X_real_te','X_syn_tr','X_syn_te']):
            file_name = os.path.join(save_path,name+'.pkl')
            with open(file_name, 'wb') as file:
                pickle.dump(data, file)
    return X_real_tr,X_real_te,X_syn_tr,X_syn_te

if __name__=='__main__':
    #------------------------------------------------------------------------------
    #LOAD DATA
    #load real and synthetic data
    path = 'C:/Users/Jim/Documents/thesis_paper/data/mimic_iv_preprocessed'
    version = 'v0.0'
    load_path = os.path.join(path,'generated',version)
    real_file = 'real_data_221223.csv'
    #syn_file = 'real_data_221223.csv'

    cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
    real_df = pd.read_csv(os.path.join(load_path,real_file),usecols=cols)

    save_path = os.path.join(path,'preprocessed',version)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #------------------------------------------------------------------------------
    #TEMP FOR TESTING
    np.random.seed(123)
    sample_size = 100
    split = np.random.choice(real_df.subject_id.unique(),int(real_df.subject_id.nunique()/2))
    d = real_df.subject_id.unique()
    split1 = np.random.choice(d,sample_size)
    d = [x for x in d if x not in split1]
    split2 = np.random.choice(d,sample_size)
    syn_df = real_df[real_df.subject_id.isin(split1)]
    real_df = real_df[real_df.subject_id.isin(split2)]

    #------------------------------------------------------------------------------
    #PREPROCESS AND SAVE DATA
    general_preprocess(real_df,syn_df,save_path)

