#script for evaluating Attribute Inference Attack on sensitive attributes
#attributes: age,gender,race

import os 
import pandas as pd
import pickle
import numpy as np
from utils import metrics,models
import keras

def privacy_AIA(X_real_tr,X_real_te,X_syn_tr,X_syn_te,syn_model,version):
    result_path = 'results/' + f'/{syn_model}/{version}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model_path = 'model/' + f'/{syn_model}/{version}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #ensure we are working on a copy of the input 
    def return_copy(df):
         return [df[0].copy(),np.copy(df[1])]
    X_real_tr = return_copy(X_real_tr)
    X_real_te = return_copy(X_real_te)
    X_syn_tr = return_copy(X_syn_tr)
    X_syn_te = return_copy(X_syn_te)

    #clear output file 
    filename = 'privacy_AIA.txt'
    with open(os.path.join(result_path,filename),'w') as f:
        pass
    #perform AIA for every label combination
    for j,labels in enumerate([['age'],['gender'],['race'],['age','gender'],['age','race'],['gender','race'],['age','gender','race']]):
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
            x.append([x0,data[1].astype(float)]) 
        x_real_tr,x_real_te,x_syn_tr,x_syn_te = x 
        y_real_tr,y_real_te,y_syn_tr,y_syn_te = y 

        #build the model, which takes account of input labels and builds output layer accordingly
        model = models.privacy_RNN(labels,nodes_at_input=100)
        #specify the loss functions and metrics we require
        #note that the output layer names are output_1 , output_2, output_3 and are dynamic not fixed
        #changing this is TBD (however does not seem possible due to dynamic structure of the model)
        losses = {}
        metric = {}
        var_count = 1
        if 'age' in labels:
            key = 'output_'+str(var_count)
            losses[key] = 'mse'
            metric[key] = 'mse'
            var_count+=1
        if 'gender' in labels:
            key = 'output_'+str(var_count)
            losses[key] = 'binary_crossentropy'
            metric[key] = 'accuracy'
            var_count+=1
        if sum(l.count('race') for l in labels)>0:
            key = 'output_'+str(var_count)
            losses[key] = 'categorical_crossentropy'
            metric[key] = 'accuracy'

        model.compile(optimizer='Adam',loss=losses,metrics=metric)

        #keras model expects a list of the different outputs instead of a concatenated array
        #also take care of turning them into float numpy arrays
        y_list = []
        for data in [y_real_tr,y_real_te,y_syn_tr,y_syn_te]:
            output_list = []
            for label in [x for x in labels if 'race' not in x]:
                output_list.append(data[label].to_numpy().astype(float))
            if sum(x.count('race') for x in labels)>0:
                output_list.append(data[[x for x in labels if 'race' in x]].to_numpy().astype(float))
            y_list.append(output_list)
        y_real_tr,y_real_te,y_syn_tr,y_syn_te = y_list
        

        #we train the model on synthetic data and assess on real test set
        #also possible: compare with real trained model to assess what the cause of bad/good prediction is
        checkpoint_filepath = model_path+f'/privacy_AIA_label{j}.keras'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        model.fit(x_syn_tr,y_syn_tr,epochs=10,batch_size=32,validation_split=.2,callbacks=[model_checkpoint_callback])
        model.load_weights(checkpoint_filepath)
        preds = model.predict(x_real_te)

        #evaluate predictions: compute mape for age, accuracy for gender and confusion matrix metrics for race
        
        with open(os.path.join(result_path,filename),'a') as f:

            f.write('Labels for this iteration: '+str(labels)+'\n')
            var_count = 0
            if 'age' in labels:
                metric = metrics.mape(y_real_te[var_count],preds[var_count])
                f.write('Age MAPE is: '+str(metric)+'\n')
                var_count+=1
            if 'gender' in labels:
                metric = metrics.accuracy(y_real_te[var_count],np.round(preds[var_count]))
                f.write('gender accuracy is: '+str(metric)+'\n')
                var_count+=1
            if sum(x.count('race') for x in labels)>0:
                metric = metrics.accuracy(np.concatenate(y_real_te[var_count:],axis=1),
                                        np.squeeze(np.round(preds[var_count:]),0))
                f.write('race accuracy is: '+str(metric)+'\n')
            f.write('\n')


if __name__=='__main__':  
#load real and synthetic data
    path = 'C:/Users/Jim/Documents/thesis_paper/data'
    version = 'v0.0'
    syn_model = 'cpar'
    
    load_path = path + '/processed' + '/preprocessed_eval' + f'/{syn_model}' + f'/{version}' 
    files = ['X_real_tr','X_real_te','X_syn_tr','X_syn_te']
    data = []
    for file in files:
        file = file+'.pkl'
        with open(os.path.join(load_path,file),'rb') as f:
            data.append(pickle.load(f))
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = data


    privacy_AIA(X_real_tr,X_real_te,X_syn_tr,X_syn_te,syn_model,version)
    
            
            
