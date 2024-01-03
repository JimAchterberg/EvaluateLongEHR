import os
import pandas as pd
import pickle
import numpy as np
from utils import preprocess,utility,fidelity

def GoF(X_real_tr,X_real_te,X_syn_tr,X_syn_te,result_path):
        #data selection
        X_train = [pd.concat([X_real_tr[0],X_syn_tr[0]],axis=0),\
                np.concatenate((X_real_tr[1],X_syn_tr[1]),axis=0)]
        X_test = [pd.concat([X_real_te[0],X_syn_te[0]],axis=0),\
                np.concatenate((X_real_te[1],X_syn_te[1]),axis=0)]
        y_train = np.concatenate((np.zeros(X_real_tr[0].shape[0]),np.ones(X_syn_tr[0].shape[0])),axis=0)
        y_test = np.concatenate((np.zeros(X_real_te[0].shape[0]),np.ones(X_syn_te[0].shape[0])),axis=0)
        data_list = [X_train[0],X_train[1],X_test[0],X_test[1],y_train,y_test]
        data_list = [np.array(obj) if isinstance(obj, pd.DataFrame) else obj for obj in data_list]
        data_list = [arr.astype(float) for arr in data_list]
        X_train[0],X_train[1],X_test[0],X_test[1],y_train,y_test = data_list

        #fit a keras model and perform GoF test
        model = fidelity.gof_model()
        model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='accuracy')
        model.fit(X_train,y_train,batch_size=32,epochs=1,validation_split=.2)
        pred = model.predict(X_test)
        test_stat,pval = fidelity.ks_test(real_pred=pred[y_test==0],syn_pred=pred[y_test==1])

        #additional numbers for final report
        accuracy = fidelity.accuracy(y_test,np.round(pred))
        total_real = y_test.shape[0]-sum(y_test)
        total_syn = sum(y_test)
        correct_real = np.sum((y_test.flatten()==np.round(pred).flatten())[y_test.flatten()==0])
        correct_syn = np.sum((y_test.flatten()==np.round(pred).flatten())[y_test.flatten()==1])

        #make a report of results
        filename = 'gof_test_report.txt'
        with open(os.path.join(result_path,filename),'w') as f:
            f.write('accuracy: ' + str(accuracy) + '\n')
            f.write('correct real: ' + str(correct_real) + '\n')
            f.write('correct synthetic: ' + str(correct_syn) + '\n')
            f.write('total real: ' + str(total_real) + '\n')
            f.write('total synthetic: ' + str(total_syn) + '\n')
            f.write('p-value: ' + str(pval) + '\n')

def trajectory_prediction(X_real_tr,X_real_te,X_syn_tr,X_syn_te,result_path):
        #get input/ouput pairs
        max_t = X_real_tr[1].shape[1]
        x = []
        y = []
        for data in [X_real_tr,X_real_te,X_syn_tr,X_syn_te]:
            x_,y_ = utility.trajectory_input_output(data)
            x.append(x_)
            y.append(y_)
        X_real_tr,X_real_te,X_syn_tr,X_syn_te = x
        y_real_tr,y_real_te,y_syn_tr,y_syn_te = y
        #turn all data into float numpy arrays
        data_list = [X_real_tr[0],X_real_tr[1],X_real_te[0],X_real_te[1],\
                    X_syn_tr[0],X_syn_tr[1],X_syn_te[0],X_syn_te[1],\
                        y_real_tr,y_real_te,y_syn_tr,y_syn_te]
        data_list = [np.array(obj) if isinstance(obj, pd.DataFrame) else obj for obj in data_list]
        data_list = [arr.astype(float) for arr in data_list]
        X_real_tr[0],X_real_tr[1],X_real_te[0],X_real_te[1],\
                    X_syn_tr[0],X_syn_tr[1],X_syn_te[0],X_syn_te[1],\
                        y_real_tr,y_real_te,y_syn_tr,y_syn_te = data_list

        #build model and predict
        real_model = utility.trajectory_RNN_simple(output_size=y_real_tr.shape[1])
        syn_model = utility.trajectory_RNN_simple(output_size=y_syn_tr.shape[1])
        real_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics='accuracy')
        syn_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics='accuracy')
        real_model.fit(X_real_tr,y_real_tr,batch_size=32,epochs=1,validation_split=.2)
        syn_model.fit(X_syn_tr,y_syn_tr,batch_size=32,epochs=1,validation_split=.2)
        real_preds = real_model.predict(X_real_te)
        syn_preds = syn_model.predict(X_real_te)
        #evaluate results
        labels = np.argmax(y_real_te,axis=1)
        real_preds_labels = np.argmax(real_preds,axis=1)
        syn_preds_labels = np.argmax(syn_preds,axis=1)
        real_acc = utility.accuracy(labels,real_preds_labels)
        syn_acc = utility.accuracy(labels,syn_preds_labels)
        filename = 'trajectory_pred_accuracy.txt'
        with open(os.path.join(result_path,filename),'w') as f:
            f.write('Real accuracy: ' + str(real_acc) + '\n')
            f.write('Synthetic accuracy: ' + str(syn_acc) + '\n')
            
def mortality_prediction(X_real_tr,X_real_te,X_syn_tr,X_syn_te,result_path,model='RNN'):
    if model not in ['RNN','LR','RF']:
         raise Exception('please input RNN, LR or RF as model')
    
    #ensure we are working on copies as to not alter the original data
    def return_copy(df):
         return [df[0].copy(),np.copy(df[1])]
    X_real_tr = return_copy(X_real_tr)
    X_real_te = return_copy(X_real_te)
    X_syn_tr = return_copy(X_syn_tr)
    X_syn_te = return_copy(X_syn_te)
    #get input/output data
    target = ['deceased']
    x0 = []
    y = []
    
    for data in [X_real_tr[0],X_real_te[0],X_syn_tr[0],X_syn_te[0]]:
        y.append(data[target])
        x0.append(data.drop(target,axis=1))
    X_real_tr[0],X_real_te[0],X_syn_tr[0],X_syn_te[0] = x0
    y_real_tr,y_real_te,y_syn_tr,y_syn_te = y

    #turn all data into float numpy arrays
    data_list = [X_real_tr[0],X_real_tr[1],X_real_te[0],X_real_te[1],\
                X_syn_tr[0],X_syn_tr[1],X_syn_te[0],X_syn_te[1],\
                    y_real_tr,y_real_te,y_syn_tr,y_syn_te]
    data_list = [np.array(obj) if isinstance(obj, pd.DataFrame) else obj for obj in data_list]
    data_list = [arr.astype(float) for arr in data_list]
    X_real_tr[0],X_real_tr[1],X_real_te[0],X_real_te[1],\
                X_syn_tr[0],X_syn_tr[1],X_syn_te[0],X_syn_te[1],\
                    y_real_tr,y_real_te,y_syn_tr,y_syn_te = data_list
    
    if model=='RNN':
        #build model and predict
        real_model = utility.mortality_RNN_simple()
        real_model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='accuracy')
        real_model.fit(X_real_tr,y_real_tr,batch_size=32,epochs=1,validation_split=.2)
        real_preds = real_model.predict(X_real_te)
        syn_model = utility.mortality_RNN_simple()
        syn_model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='accuracy')
        syn_model.fit(X_syn_tr,y_syn_tr,batch_size=32,epochs=1,validation_split=.2)
        syn_preds = real_model.predict(X_real_te)

    elif model=='LR':
        #turn sequence data into count features and then to single array
        x = []
        for data in [X_real_tr,X_real_te,X_syn_tr,X_syn_te]:
             x0,x1 = data 
             x1 = np.sum(x1,axis=1)
             x.append(np.concatenate((x0,x1),axis=1))
        X_real_tr,X_real_te,X_syn_tr,X_syn_te = x
        real_model = utility.mortality_LR(penalty='elasticnet',l1_ratio=.5)
        real_model.fit(X_real_tr,y_real_tr.flatten())
        real_preds = real_model.predict(X_real_te)
        syn_model = utility.mortality_LR(penalty='elasticnet',l1_ratio=.5)
        syn_model.fit(X_syn_tr,y_syn_tr.flatten())
        syn_preds = syn_model.predict(X_real_te)
    else:
        #turn sequence data into count features and then to single array
        x = []
        for data in [X_real_tr,X_real_te,X_syn_tr,X_syn_te]:
             x0,x1 = data 
             x1 = np.sum(x1,axis=1)
             x.append(np.concatenate((x0,x1),axis=1))
        X_real_tr,X_real_te,X_syn_tr,X_syn_te = x
        real_model = utility.mortality_RF(n_estimators=100,max_depth=None)
        real_model.fit(X_real_tr,y_real_tr.flatten())
        real_preds = real_model.predict(X_real_te)
        syn_model = utility.mortality_RF(n_estimators=100,max_depth=None)
        syn_model.fit(X_syn_tr,y_syn_tr.flatten())
        syn_preds = syn_model.predict(X_real_te)

    #evaluate results
    real_acc = utility.accuracy(y_real_te,np.round(real_preds))
    syn_acc = utility.accuracy(y_real_te,np.round(syn_preds))
    real_auc = utility.auc(y_real_te,real_preds)
    syn_auc = utility.auc(y_real_te,syn_preds)
    class_balance = sum(y_real_te)/len(y_real_te)
    filename = model+'_'+'mortality_pred_accuracy.txt'
    with open(os.path.join(result_path,filename),'w') as f:
        f.write('Real accuracy: ' + str(real_acc) + '\n')
        f.write('Synthetic accuracy: ' + str(syn_acc) + '\n')
        f.write('Real AUC: ' + str(real_auc) + '\n')
        f.write('Synthetic AUC: ' + str(syn_auc) + '\n')
        f.write('Class balance: '+str(class_balance)+'\n')

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

    #trajectory_prediction(X_real_tr,X_real_te,X_syn_tr,X_syn_te,result_path)
    #GoF(X_real_tr,X_real_te,X_syn_tr,X_syn_te,result_path)
    #mortality_prediction(X_real_tr,X_real_te,X_syn_tr,X_syn_te,result_path,model='RNN')
    #mortality_prediction(X_real_tr,X_real_te,X_syn_tr,X_syn_te,result_path,model='RF')
    #mortality_prediction(X_real_tr,X_real_te,X_syn_tr,X_syn_te,result_path,model='LR')

    