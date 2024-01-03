#script for evaluating Attribute Inference Attack on sensitive attributes
#attributes: age,gender,race

#take preprocessed data 
#use all different combinations of the three sensitive attributes as labels:
# (age), (gender), (race), (age,gender), (age,race), (gender,race), (age,gender,race)
#use all non-label data as input

import os 
import pandas as pd
import pickle
import numpy as np

if __name__=='__main__':  
#load real and synthetic data
    path = 'C:/Users/Jim/Documents/thesis_paper/data/mimic_iv_preprocessed'
    version = 'v0.0'
    load_path = os.path.join(path,'preprocessed',version)
    result_path = os.path.join('results',version)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    files = ['X_real_tr','X_real_te','X_syn_tr','X_syn_te']
    data = []
    for file in files:
        file = file+'.pkl'
        with open(os.path.join(load_path,file),'rb') as f:
            data.append(pickle.load(f))
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = data[0],data[1],data[2],data[3]

    #ensure we are working on a copy of the input 
    def return_copy(df):
         return [df[0].copy(),np.copy(df[1])]
    X_real_tr = return_copy(X_real_tr)
    X_real_te = return_copy(X_real_te)
    X_syn_tr = return_copy(X_syn_tr)
    X_syn_te = return_copy(X_syn_te)
    #perform AIA for every label combination
    for labels in [['age'],['gender'],['race'],['age','gender'],['age','race'],['gender','race'],['age','gender','race']]:
        #ensure we are taking the one hot encoded columns
        if 'race' in labels:
            labels.remove('race')
            labels = labels + [x for x in X_real_tr[0].columns if 'race' in x] 
        y = []
        x = []
        for data in [X_real_tr,X_real_te,X_syn_tr,X_syn_te]:
            #find target data
            y.append(data[0][labels])
            #find input data
            x.append([data[0].drop(labels,axis=1),data[1]]) 
        x_real_tr,x_real_te,x_syn_tr,x_syn_te = x 
        y_real_tr,y_real_te,y_syn_tr,y_syn_te = y 

        # #build model and make predictions 

        # #infer metadata: amount of label columns for output node size, which are categorical/numerical for softmax/linear output node 