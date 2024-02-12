import os
import pandas as pd
import pickle
import numpy as np
from utils import preprocess,metrics,models
import keras


def GoF(data,hparams,syn_model):
        X_real_tr,X_real_te,X_syn_tr,X_syn_te = data
        model_path = os.path.join('model',syn_model)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        #data selection
        X_train = [np.concatenate((X_real_tr[0],X_syn_tr[0]),axis=0),\
                np.concatenate((X_real_tr[1],X_syn_tr[1]),axis=0)]
        X_test = [np.concatenate((X_real_te[0],X_syn_te[0]),axis=0),\
                np.concatenate((X_real_te[1],X_syn_te[1]),axis=0)]
        y_train = np.concatenate((np.zeros(X_real_tr[0].shape[0]),np.ones(X_syn_tr[0].shape[0])),axis=0)
        y_test = np.concatenate((np.zeros(X_real_te[0].shape[0]),np.ones(X_syn_te[0].shape[0])),axis=0)

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
        #save plot of distribution of real/synthetic predictions
        GoF_kdeplot = metrics.GoF_kdeplot(pred=pred,y_test=y_test)
        GoF_kdeplot.title('Distribution of classification scores')


        return accuracy,pval,GoF_kdeplot

def mortality_prediction(data,syn_model,hparams,pred_model='RNN'):
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = data
    model_path = os.path.join('model',syn_model)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    assert pred_model  in ['RNN','LR','RF']
    #get input/output data
    target = 'deceased'
    target = attr.get_loc(target)
    x0 = []
    y = []
    for data in [X_real_tr[0],X_real_te[0],X_syn_tr[0],X_syn_te[0]]:
        y.append(data[:,target])
        x0.append(np.delete(data,target,axis=1))
    X_real_tr[0],X_real_te[0],X_syn_tr[0],X_syn_te[0] = x0
    y_real_tr,y_real_te,y_syn_tr,y_syn_te = y
    
    if pred_model=='RNN':
        model_list = []
        for data,name in zip([[X_real_tr,y_real_tr],[X_syn_tr,y_syn_tr]],['real','syn']):
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
    else:
        x = []
        for data in [X_real_tr,X_real_te,X_syn_tr,X_syn_te]:
             x0,x1 = data 
             x1 = np.sum(x1,axis=1)
             x.append(np.concatenate((x0,x1),axis=1))
        X_real_tr,X_real_te,X_syn_tr,X_syn_te = x
        model_list = []
        for data,name in zip([[X_real_tr,y_real_tr],[X_syn_tr,y_syn_tr]],['real','syn']):
            X_tr,y_tr = data
            checkpoint_filepath=model_path + f'/mortality_{pred_model}_{name}'+'.pkl'
            if pred_model=='LR':
                model = models.mortality_LR(penalty='elasticnet',l1_ratio=hparams['L1'])
            else:
                model = models.mortality_RF(n_estimators=hparams['N_TREES'],max_depth=hparams['MAX_DEPTH'])
            model.fit(X_tr,y_tr.flatten())
            with open(checkpoint_filepath,'wb') as f:
                pickle.dump(model,f)
            model_list.append(model)
        real_model,syn_model = model_list
    #make predictions
    real_preds = real_model.predict(X_real_te)
    syn_preds = syn_model.predict(X_real_te)
    real_acc = metrics.accuracy(y_real_te,np.round(real_preds))
    syn_acc = metrics.accuracy(y_real_te,np.round(syn_preds))
    real_auc = metrics.auc(y_real_te,real_preds)
    syn_auc = metrics.auc(y_real_te,syn_preds)
    return real_acc,real_auc,syn_acc,syn_auc

def trajectory_prediction(data,hparams,syn_model):
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = data
    model_path = os.path.join('model',syn_model)
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
    return real_acc,syn_acc

            
    

    
if __name__=='__main__':  
    path = 'C:/Users/Jim/Documents/thesis_paper'
    syn_model = 'cpar'
    result_path = os.path.join('results',syn_model)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #------------------------------------------------------------------------------------------
    #LOAD DATA
    cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
    real_df = pd.read_csv(path+'/real.csv.gz',sep=',',compression='gzip',usecols=cols)
    syn_df = pd.read_csv(path+'/cpar.csv.gz',sep=',',compression='gzip',usecols=cols)

    #select only part of data for testing
    n = 100
    real_df = real_df[real_df.subject_id.isin(np.random.choice(real_df.subject_id.unique(),size=n))]
    syn_df = syn_df[syn_df.subject_id.isin(np.random.choice(syn_df.subject_id.unique(),size=n))]
    #------------------------------------------------------------------------------------------
    #PREPROCESS
    #takes real and syn data and outputs encoded list of static and sequential data
    real,syn = preprocess.prepr(real_df,syn_df)
    real[0] = real[0].reset_index(drop=True)
    syn[0] = syn[0].reset_index(drop=True)
    attr = real[0].columns #necessary to check target column later

    #make k test splits
    idx = np.arange(0,real[0].shape[0]) #find full data indices to select splits from
    k = 3   #number of splits
    test_size = .3 #test split percentage
    test_splits = [np.random.choice(idx, size=int(test_size*real[0].shape[0]), replace=False) for _ in range(k)]
    #------------------------------------------------------------------------------------------
    #LOOP OVER TRAIN TEST SPLITS
    #setup report files for later appending
    filename = 'gof_test_report.txt'
    with open(os.path.join(result_path,filename),'w') as f:
        pass
    filename = 'mortality_pred_accuracy.txt'
    with open(os.path.join(result_path,filename),'w') as f:
        pass
    filename = 'trajectory_pred_accuracy.txt'
    with open(os.path.join(result_path,filename),'w') as f:
        pass
    s=0 #indicates at which split we are
    for te in test_splits:
        tr = np.setdiff1d(idx,te)
        #select train test data and scale numericals per dataset to avoid information leakage
        age_scaler = preprocess.Scaler()
        age_scaler.transform(real[0].loc[te]['age']) #save for later reverse transformation on test data
        list_ = []
        for data in [real,syn]:
            for i in [tr,te]:
                stat = data[0].loc[i]
                stat['age'] = preprocess.Scaler().transform(stat['age'])
                stat = stat.to_numpy().astype(float)
                seq = data[1][i]
                seq = seq.astype(float)
                list_.append([stat,seq])
        #------------------------------------------------------------------------------------------
        #GoF
        nn_params = {'EPOCHS':10,
               'BATCH_SIZE':16,
               'HIDDEN_UNITS':[100],
               'ACTIVATION':'relu',
               'DROPOUT_RATE':.2
               }
        acc,pval,plot = GoF(data=list_,hparams=nn_params,syn_model='cpar')
        filename = 'gof_test_report.txt'
        with open(os.path.join(result_path,filename),'a') as f:
            f.write(f'accuracy at fold {s}: {str(acc)}'+'\n')
            f.write(f'pval at fold {s}: {str(pval)}'+'\n')
        #save plot only at even numbers
        if (s%2)==0:
            filename = f'gof_plot_{s}.png'
            plot.savefig(os.path.join(result_path,filename))
        #------------------------------------------------------------------------------------------
        #mortality 
        pred_model = 'RNN'
        nn_params = {'EPOCHS':10,
               'BATCH_SIZE':16,
               'HIDDEN_UNITS':[100],
               'ACTIVATION':'relu',
               'DROPOUT_RATE':.2
               }
        real_acc,real_auc,syn_acc,syn_auc = mortality_prediction(data=list_,syn_model=syn_model,hparams=nn_params,pred_model=pred_model)
        filename = 'mortality_pred_accuracy.txt'
        with open(os.path.join(result_path,filename),'a') as f:
            f.write(f'{pred_model} Real accuracy at fold {s}: {real_acc}'+'\n')
            f.write(f'{pred_model} Synthetic accuracy at fold {s}: {syn_acc}'+'\n')
            f.write(f'{pred_model} Real AUC at fold {s}: {real_auc}'+'\n')
            f.write(f'{pred_model} Synthetic AUC at fold {s}: {syn_auc}'+'\n')
        #------------------------------------------------------------------------------------------
        #trajectory prediction
        nn_params = {'EPOCHS':10,
            'BATCH_SIZE':16,
            'HIDDEN_UNITS':[100],
            'ACTIVATION':'relu',
            'DROPOUT_RATE':.2
            }
        real_acc,syn_acc = trajectory_prediction(data=list_,hparams=nn_params,syn_model=syn_model)
        filename = 'trajectory_pred_accuracy.txt'
        with open(os.path.join(result_path,filename),'a') as f:
            f.write(f'Real accuracy at fold {s}: {real_acc}'+'\n')
            f.write(f'Synthetic accuracy at fold {s}: {syn_acc}'+'\n')
        s+=1


    # nn_params = {'EPOCHS':100,
    #            'BATCH_SIZE':128,
    #            'HIDDEN_UNITS':[100],
    #            'ACTIVATION':'relu',
    #            'DROPOUT_RATE':.2
    #            }
    # rf_params = {'N_TREES':100,
    #              'MAX_DEPTH':5}
    # lr_params = {'L1':1} 

   

    