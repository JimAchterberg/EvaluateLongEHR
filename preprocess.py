
import pandas as pd
import os
import numpy as np
from utils import preprocess
import pickle

# #preprocess data for evaluation
def preprocess_eval(real_df,syn_df):
    #ONE HOT ENCODE
    df = pd.concat([real_df,syn_df],axis=0)
    for col in ['race','icd_code']:
        dummies = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,dummies],axis=1)
        df = df.drop(col,axis=1)
    n_real = real_df.subject_id.nunique()
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
            X.append([data[0].iloc[idx],data[1][idx]])
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = X

    return X_real_tr,X_real_te,X_syn_tr,X_syn_te

if __name__=='__main__':
    #------------------------------------------------------------------------------
    #LOAD DATA
    #load real and synthetic data
    path = 'C:/Users/Jim/Documents/thesis_paper/data'
    version = 'v0.0'
    model = 'dgan'
    load_path = path + '/processed' + '/generated'

    cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
    real_df = pd.read_csv(load_path+'/real'+'/real.csv.gz',sep=',',compression='gzip',usecols=cols)
    syn_df = pd.read_csv(load_path+f'/{model}'+f'/{model}_{version}.csv.gz',sep=',',compression='gzip',usecols=cols)

    
    #------------------------------------------------------------------------------
    #preprocess real and synthetic data to train and test sets
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = preprocess_eval(real_df,syn_df)
    #directory for saving preprocessed data
    save_path = path + '/processed' + '/preprocessed_eval' + f'/{model}' + f'/{version}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #save data as pickle objects
    data_list = [X_real_tr,X_real_te,X_syn_tr,X_syn_te]
    files = ['X_real_tr','X_real_te','X_syn_tr','X_syn_te']
    
    for data,name in zip(data_list,files):
        file_name = os.path.join(save_path,name+'.pkl')
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
    

