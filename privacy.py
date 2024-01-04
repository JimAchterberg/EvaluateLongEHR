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
from utils import privacy

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


    #-----------------------------------------------------------------------------------
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
            x0 = data[0].drop(labels,axis=1)
            x0 = x0.to_numpy().astype(float)
            x.append([x0,data[1]]) 
        x_real_tr,x_real_te,x_syn_tr,x_syn_te = x 
        y_real_tr,y_real_te,y_syn_tr,y_syn_te = y 

        #build the model, which takes account of input labels and builds output layer accordingly
        model = privacy.privacy_RNN(labels,nodes_at_input=100)
        #specify the loss functions and metrics we require
        #note that the output layer names are output_1:age, output_2:gender, output_3:race
        #changing this is TBD
        losses = {}
        metrics = {}
        if 'age' in labels:
            losses['output_1'] = 'mse'
            metrics['output_1'] = 'mse'
        if 'gender' in labels:
            losses['output_2'] = 'binary_crossentropy'
            metrics['output_2'] = 'accuracy'
        if sum(l.count('race') for l in labels)>0:
            losses['output_3'] = 'categorical_crossentropy'
            metrics['output_3'] = 'accuracy'

        model.compile(optimizer='Adam',loss=losses,metrics=metrics)

        #keras model expects a list of the different outputs instead of a concatenated array
        #also take care of turning them into float numpy arrays
        y_list = []
        for data in [y_real_tr,y_real_te,y_syn_tr,y_real_tr]:
            output_list = []
            for label in [x for x in labels if 'race' not in x]:
                output_list.append(data[label].to_numpy().astype(float))
            if sum(x.count('race') for x in labels)>0:
                output_list.append(data[[x for x in labels if 'race' in x]].to_numpy().astype(float))
            y_list.append(output_list)
        y_real_tr,y_real_te,y_syn_tr,y_real_tr = y_list

        #we train the model on synthetic data and assess on real test set
        #also possible: compare with real trained model to assess what the cause of bad/good prediction is
        model.fit(x_syn_tr,y_syn_tr,epochs=1,batch_size=32,validation_split=.2)
        preds = model.predict(x_real_te)

        #evaluate predictions: compute mape for age, accuracy for gender and confusion matrix metrics for race
