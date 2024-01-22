import os
import pandas as pd
import pickle
import numpy as np
from utils import preprocess,metrics,models
import keras


def GoF(data,hparams,syn_model,version):
        X_real_tr,X_real_te,X_syn_tr,X_syn_te = data
        result_path = os.path.join('results',syn_model,version)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        model_path = os.path.join('model',syn_model,version)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        #data selection
        X_train = [pd.concat([X_real_tr[0],X_syn_tr[0]],axis=0),\
                np.concatenate((X_real_tr[1],X_syn_tr[1]),axis=0)]
        X_test = [pd.concat([X_real_te[0],X_syn_te[0]],axis=0),\
                np.concatenate((X_real_te[1],X_syn_te[1]),axis=0)]
        y_train = np.concatenate((np.zeros(X_real_tr[0].shape[0]),np.ones(X_syn_tr[0].shape[0])),axis=0)
        y_test = np.concatenate((np.zeros(X_real_te[0].shape[0]),np.ones(X_syn_te[0].shape[0])),axis=0)
        #zero one scale age
        X_train[0]['age'] = preprocess.Scaler().transform(X_train[0]['age'])
        X_test[0]['age'] = preprocess.Scaler().transform(X_test[0]['age'])
        #turn everything into float numpy arrays
        data_list = [X_train[0],X_train[1],X_test[0],X_test[1],y_train,y_test]
        data_list = [np.array(obj) if isinstance(obj, pd.DataFrame) else obj for obj in data_list]
        data_list = [arr.astype(float) for arr in data_list]
        X_train[0],X_train[1],X_test[0],X_test[1],y_train,y_test = data_list

        #set GoF Keras model up for checkpoint saving best model
        checkpoint_filepath = model_path + '/GoF' + '.keras'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
        #instantiate and fit model
        config = {'input_shape_attr':(X_train[0].shape[1],),
                  'input_shape_feat':(X_train[1].shape[1],X_train[1].shape[2],),
                  #first layer is separate processing, afterwards joint Dense layers
                  'hidden_units':hparams['HIDDEN_UNITS'],
                  'dropout_rate':hparams['DROPOUT_RATE'],
                  'activation':hparams['ACTIVATION']
                  }
        model = models.GoF_RNN(config=config)
        model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='accuracy')
        model.fit(X_train,y_train,batch_size=hparams['BATCH_SIZE'],epochs=hparams['EPOCHS'],
                  validation_split=.2,callbacks=[model_checkpoint_callback])
        #load best model and make predictions
        model.load_weights(checkpoint_filepath)
        pred = model.predict(X_test)
        test_stat,pval = metrics.ks_test(real_pred=pred[y_test==0],syn_pred=pred[y_test==1])

        #additional numbers for final report
        accuracy = metrics.accuracy(y_test,np.round(pred))
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

        #save plot of distribution of real/synthetic predictions
        GoF_kdeplot = metrics.GoF_kdeplot(pred=pred,y_test=y_test)
        GoF_kdeplot.title('Distribution of classification scores')
        filename = 'GoF_kdeplot.png'
        GoF_kdeplot.savefig(os.path.join(result_path,filename))

def trajectory_prediction(data,hparams,syn_model,version):
        X_real_tr,X_real_te,X_syn_tr,X_syn_te = data
        result_path = os.path.join('results',syn_model,version)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        model_path = os.path.join('model',syn_model,version)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        x = []
        y = []
        for data in [X_real_tr,X_real_te,X_syn_tr,X_syn_te]:
            x_,y_ = preprocess.trajectory_input_output(data,max_t=X_real_tr[1].shape[1])
            x.append(x_)
            y.append(y_)
        x_real_tr,x_real_te,x_syn_tr,x_syn_te = x
        y_real_tr,y_real_te,y_syn_tr,y_syn_te = y
        #zero one scale age
        x_real_tr[0]['age'] = preprocess.Scaler().transform(x_real_tr[0]['age'])
        x_real_te[0]['age'] = preprocess.Scaler().transform(x_real_te[0]['age'])
        x_syn_tr[0]['age'] = preprocess.Scaler().transform(x_syn_tr[0]['age'])
        x_syn_te[0]['age'] = preprocess.Scaler().transform(x_syn_te[0]['age'])
        #turn all data into float numpy arrays
        data_list = [x_real_tr[0],x_real_tr[1],x_real_te[0],x_real_te[1],\
                    x_syn_tr[0],x_syn_tr[1],x_syn_te[0],x_syn_te[1],\
                        y_real_tr,y_real_te,y_syn_tr,y_syn_te]
        data_list = [np.array(obj) if isinstance(obj, pd.DataFrame) else obj for obj in data_list]
        data_list = [arr.astype(float) for arr in data_list]
        x_real_tr[0],x_real_tr[1],x_real_te[0],x_real_te[1],\
                    x_syn_tr[0],x_syn_tr[1],x_syn_te[0],x_syn_te[1],\
                        y_real_tr,y_real_te,y_syn_tr,y_syn_te = data_list
        
        #fit the synthetic and real model
        model_list = []
        for data,name in zip([[x_real_tr,y_real_tr],[x_syn_tr,y_syn_tr]],['real','syn']):
            X_tr,y_tr = data
            checkpoint_filepath=model_path + f'/trajectory_{name}'+'.keras'
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
            config = {'input_shape_attr':(X_tr[0].shape[1],),
                  'input_shape_feat':(X_tr[1].shape[1],X_tr[1].shape[2],),
                  #first layer is separate processing, afterwards joint Dense layers
                  'hidden_units':hparams['HIDDEN_UNITS'],
                  'dropout_rate':hparams['DROPOUT_RATE'],
                  'activation':hparams['ACTIVATION'],
                  'output_units':y_tr.shape[1]
                  }
            model = models.trajectory_RNN_simple(config=config)
            model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics='accuracy')
            model.fit(X_tr,y_tr,batch_size=hparams['BATCH_SIZE'],epochs=hparams['EPOCHS'],
                      validation_split=.2,callbacks=[model_checkpoint_callback])
            model.load_weights(checkpoint_filepath)
            model_list.append(model)
        real_model,syn_model = model_list
        #make predictions
        real_preds = real_model.predict(x_real_te)
        syn_preds = syn_model.predict(x_real_te)
        #evaluate results
        labels = np.argmax(y_real_te,axis=1)
        real_preds_labels = np.argmax(real_preds,axis=1)
        syn_preds_labels = np.argmax(syn_preds,axis=1)
        real_acc = metrics.accuracy(labels,real_preds_labels)
        syn_acc = metrics.accuracy(labels,syn_preds_labels)
        filename = 'trajectory_pred_accuracy.txt'
        with open(os.path.join(result_path,filename),'w') as f:
            f.write('Real accuracy: ' + str(real_acc) + '\n')
            f.write('Synthetic accuracy: ' + str(syn_acc) + '\n')
            
def mortality_prediction(syn_model,version,hparams,pred_model='RNN'):
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = data
    result_path = os.path.join('results',syn_model,version)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model_path = os.path.join('model',syn_model,version)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    assert pred_model  in ['RNN','LR','RF']
    #ensure we are working on copies as to not alter the original data
    def return_copy(df):
         return [df[0].copy(),np.copy(df[1])]
    x_real_tr = return_copy(X_real_tr)
    x_real_te = return_copy(X_real_te)
    x_syn_tr = return_copy(X_syn_tr)
    x_syn_te = return_copy(X_syn_te)
    #get input/output data
    target = ['deceased']
    x0 = []
    y = []
    for data in [x_real_tr[0],x_real_te[0],x_syn_tr[0],x_syn_te[0]]:
        y.append(data[target])
        x0.append(data.drop(target,axis=1))
    x_real_tr[0],x_real_te[0],x_syn_tr[0],x_syn_te[0] = x0
    y_real_tr,y_real_te,y_syn_tr,y_syn_te = y
    #zero one scale age 
    x_real_tr[0]['age'] = preprocess.Scaler().transform(x_real_tr[0]['age'])
    x_real_te[0]['age'] = preprocess.Scaler().transform(x_real_te[0]['age'])
    x_syn_tr[0]['age'] = preprocess.Scaler().transform(x_syn_tr[0]['age'])
    x_syn_te[0]['age'] = preprocess.Scaler().transform(x_syn_te[0]['age'])
    #turn all data into float numpy arrays
    data_list = [x_real_tr[0],x_real_tr[1],x_real_te[0],x_real_te[1],\
                x_syn_tr[0],x_syn_tr[1],x_syn_te[0],x_syn_te[1],\
                    y_real_tr,y_real_te,y_syn_tr,y_syn_te]
    data_list = [np.array(obj) if isinstance(obj, pd.DataFrame) else obj for obj in data_list]
    data_list = [arr.astype(float) for arr in data_list]
    x_real_tr[0],x_real_tr[1],x_real_te[0],x_real_te[1],\
                x_syn_tr[0],x_syn_tr[1],x_syn_te[0],x_syn_te[1],\
                    y_real_tr,y_real_te,y_syn_tr,y_syn_te = data_list
    
    if pred_model=='RNN':
        model_list = []
        for data,name in zip([[x_real_tr,y_real_tr],[x_syn_tr,y_syn_tr]],['real','syn']):
            X_tr,y_tr = data 
            checkpoint_filepath=model_path + f'/mortality_{pred_model}_{name}'+'.keras'
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
            config = {'input_shape_attr':(X_tr[0].shape[1],),
                  'input_shape_feat':(X_tr[1].shape[1],X_tr[1].shape[2],),
                  #first layer is separate processing, afterwards joint Dense layers
                  'hidden_units':hparams['HIDDEN_UNITS'],
                  'dropout_rate':hparams['DROPOUT_RATE'],
                  'activation':hparams['ACTIVATION']
                  }
            model = models.mortality_RNN_simple(config=config)
            model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='accuracy')
            model.fit(X_tr,y_tr,batch_size=hparams['BATCH_SIZE'],epochs=hparams['EPOCHS'],
                      validation_split=.2,callbacks=[model_checkpoint_callback])
            #choose the best version and load its weights for real and synthetic model
            model.load_weights(checkpoint_filepath)
            model_list.append(model)
        real_model,syn_model = model_list
        
        real_preds = real_model.predict(x_real_te)
        syn_preds = real_model.predict(x_real_te)

    elif pred_model=='LR':
        #turn sequence data into count features and then to single array
        x = []
        for data in [x_real_tr,x_real_te,x_syn_tr,x_syn_te]:
             x0,x1 = data 
             x1 = np.sum(x1,axis=1)
             x.append(np.concatenate((x0,x1),axis=1))
        x_real_tr,x_real_te,x_syn_tr,x_syn_te = x
        model_list = []
        for data,name in zip([[x_real_tr,y_real_tr],[x_syn_tr,y_syn_tr]],['real','syn']):
            X_tr,y_tr = data
            checkpoint_filepath=model_path + f'/mortality_{pred_model}_{name}'+'.pkl'
            model = models.mortality_LR(penalty='elasticnet',l1_ratio=hparams['L1'])
            model.fit(X_tr,y_tr.flatten())
            with open(checkpoint_filepath,'wb') as f:
                pickle.dump(model,f)
            model_list.append(model)
        real_model,syn_model = model_list
            
        real_preds = real_model.predict(x_real_te)
        syn_preds = syn_model.predict(x_real_te)
    else:
        #turn sequence data into count features and then to single array
        x = []
        for data in [x_real_tr,x_real_te,x_syn_tr,x_syn_te]:
             x0,x1 = data 
             x1 = np.sum(x1,axis=1)
             x.append(np.concatenate((x0,x1),axis=1))
        x_real_tr,x_real_te,x_syn_tr,x_syn_te = x
        model_list = []
        for data,name in zip([[x_real_tr,y_real_tr],[x_syn_tr,y_syn_tr]],['real','syn']):
            X_tr,y_tr = data
            checkpoint_filepath=model_path + f'/mortality_{pred_model}_{name}'+'.pkl'
            model = models.mortality_RF(n_estimators=hparams['N_TREES'],max_depth=hparams['MAX_DEPTH'])
            model.fit(X_tr,y_tr.flatten())
            with open(checkpoint_filepath,'wb') as f:
                pickle.dump(model,f)
            model_list.append(model)
        real_model,syn_model = model_list
        #build model and predict
        real_preds = real_model.predict(x_real_te)
        syn_preds = syn_model.predict(x_real_te)

    #evaluate results
    filename = pred_model+'_'+'mortality_pred_accuracy.txt'
    #create and clear result file
    with open(os.path.join(result_path,filename),'w') as f:
        pass
    for preds,name in zip([real_preds,syn_preds],['Real','Synthetic']):
        acc = metrics.accuracy(y_real_te,np.round(preds))
        auc = metrics.auc(y_real_te,preds)
        with open(os.path.join(result_path,filename),'a') as f:
            f.write(f'{name} accuracy: '+str(acc)+'\n')
            f.write(f'{name} AUC: '+str(auc)+'\n')
            f.write('Class balance: ' + str(sum(y_real_te)/len(y_real_te))+'\n')

    
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

    nn_params = {'EPOCHS':1000,
               'BATCH_SIZE':128,
               'HIDDEN_UNITS':[100,100,50,30],
               'ACTIVATION':'relu',
               'DROPOUT_RATE':.2
               }
    rf_params = {'N_TREES':100,
                 'MAX_DEPTH':None}
    lr_params = {'L1':.5}

    GoF(data=data,syn_model=syn_model,version=version,hparams=nn_params)
    trajectory_prediction(data=data,syn_model=syn_model,version=version,hparams=nn_params)
    mortality_prediction(data=data,syn_model=syn_model,version=version,pred_model='RNN',hparams=nn_params)
    mortality_prediction(data=data,syn_model=syn_model,version=version,pred_model='RF',hparams=rf_params)
    mortality_prediction(data=data,syn_model=syn_model,version=version,pred_model='LR',hparams=lr_params)

    