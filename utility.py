import os 
import numpy as np
import pandas as pd
from utils import preprocess,utility
if __name__=='__main__':
    #load real and synthetic data
    path = 'C:/Users/Jim/Documents/thesis_paper/data/mimic_iv_preprocessed'
    file = 'real_data_221223.csv'
    version = 'v0.0'
    cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
    real_df = pd.read_csv(os.path.join(path,file),usecols=cols)

    result_path = os.path.join('results',version)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #REMOVE LATER!!!!!!!!!! ONLY NECESSARY FOR TESTING
    np.random.seed(123)
    sample_size = 100
    split = np.random.choice(real_df.subject_id.unique(),int(real_df.subject_id.nunique()/2))
    d = real_df.subject_id.unique()
    split1 = np.random.choice(d,sample_size)
    d = [x for x in d if x not in split1]
    split2 = np.random.choice(d,sample_size)
    syn_df = real_df[real_df.subject_id.isin(split1)]
    real_df = real_df[real_df.subject_id.isin(split2)]


    X = []
    Y = []
    for data in [real_df,syn_df]:
        #create input and output data
        seq = []
        sbj = []
        y = []
        seqs,sbjs = preprocess.get_sequences(data,column='icd_code',return_subject_idx=True)
        for sequence,subject_idx in zip(seqs,sbjs):
            for t in range(len(sequence)-1):
                seq.append(sequence[:t+1])
                sbj.append(subject_idx)
                y.append(sequence[t+1])

        #merge sequence and static input data to single dataframe
        seq_df = pd.DataFrame([sbj,seq,y],index=['subject_id','icd_code','y']).T
        static = preprocess.get_static(data,columns=['age','gender','deceased','race']) 
        static = static.reset_index()
        data = seq_df.merge(static,on='subject_id',how='left')
        y = data.y
        data = data.drop('y',axis=1)
        #make train test split
        train,test,y_train,y_test = preprocess.train_split(data,y,train_size=.7)

        #perform preprocessing separately for train and test sets
        for data,y in zip([train,test],[y_train,y_test]):
            #preprocess output and add to datalist
            y = pd.DataFrame(y,columns=['y'])
            y = preprocess.one_hot_encoding(y,columns=['y'],column_sizes=[119])
            Y.append(y.to_numpy().astype(float))
            #turn sequences back into lists and preprocess
            seq = data.icd_code.to_list()
            max_sequence_length = max(len(s) for s in seq)
            seq = preprocess.sequences_to_3d(seq,maxlen=max_sequence_length,padding=-1)
            seq = preprocess.one_hot_3d(seq,119+1)
            seq = seq[:,:,1:]
            #preprocess the attributes
            attributes = data[['age','gender','deceased','race']]
            attributes = preprocess.one_hot_encoding(attributes,columns=['race'],column_sizes=[6])
            attributes[['age']] = preprocess.zero_one_scale(attributes[['age']])
            X.append([attributes.to_numpy().astype(float),seq.astype(float)])
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = X[0],X[1],X[2],X[3]
    Y_real_tr,Y_real_te,Y_syn_tr,Y_syn_te = Y[0],Y[1],Y[2],Y[3]
    #build model and predict
    real_model = utility.trajectory_RNN_simple(output_size=119)
    syn_model = utility.trajectory_RNN_simple(output_size=119)
    real_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics='accuracy')
    syn_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics='accuracy')
    real_model.fit(X_real_tr,Y_real_tr,batch_size=32,epochs=1,validation_split=.2)
    syn_model.fit(X_syn_tr,Y_syn_tr,batch_size=32,epochs=1,validation_split=.2)
    real_preds = real_model.predict(X_real_te)
    syn_preds = syn_model.predict(X_real_te)
    #evaluate results
    labels = np.argmax(Y_real_te,axis=1)
    real_preds_labels = np.argmax(real_preds,axis=1)
    syn_preds_labels = np.argmax(syn_preds,axis=1)
    real_acc = utility.accuracy(labels,real_preds_labels)
    syn_acc = utility.accuracy(labels,syn_preds_labels)
    print('real_acc: ',real_acc)
    print('syn acc: ',syn_acc)
    # # preprocess for mortality model
    # # build mortality model
    # # build train/predict and evaluate script
   

    




    
    
    
    
    
    



