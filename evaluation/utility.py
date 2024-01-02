import os 
import numpy as np
import pandas as pd
from utils import preprocess,utility

def trajectory_prediction(real_df,syn_df,result_path):
    print('preprocessing for trajectory prediction...')
    #one hot encode together before train test splitting
    df = pd.concat([real_df,syn_df],axis=0)
    for col in ['race','icd_code']:
        dummies = pd.get_dummies(df[col],prefix=col)
        df = df.drop(col,axis=1)
        df = pd.concat([df,dummies],axis=1)
    real_df = df[:real_df.shape[0]]
    syn_df = df[real_df.shape[0]:]
    real_df = real_df.reset_index()
    syn_df = syn_df.reset_index()
    max_t = df.seq_num.max()
    X = []
    Y = []
    for data in [real_df,syn_df]:
        #initialize dataframes for storing input/output pairs
        seqs = []
        outputs = []
        stat = []
        #find input output pairs per timestep and add to dataframes
        for t in range(1,max_t):
            # #select data from subjects with at least t=t+1 timesteps
            data = data.groupby('subject_id').filter(lambda x: max(x.seq_num)>=t+1)
            #select input and output
            input = data[data.seq_num<=t]
            static = preprocess.get_static(input,[x for x in df.columns if 'race' in x or x in ['age','gender','deceased']]).astype(float)
            stat.append(static)
            seq = preprocess.df_to_3d(input,[x for x in df.columns if 'icd_code' in x],padding=0,pad_to=max_t).astype(float)
            seqs.append(seq)
            output = data[[x for x in df.columns if 'icd_code' in x]][data.seq_num==t+1]
            outputs.append(output)
        outputs = pd.concat(outputs,ignore_index=True)
        outputs = outputs.reset_index(drop=True)
        stat = pd.concat(stat,ignore_index=True)
        stat = stat.reset_index(drop=True)
        seqs = np.concatenate(seqs)
        #make train test split and do last preprocessing
        train_idx,test_idx,y_train,y_test = preprocess.train_split(np.arange(outputs.shape[0]),outputs,train_size=.7)#,stratify=outputs)
        Y.append(y_train)
        Y.append(y_test)

        for idx in [train_idx,test_idx]:
            x_stat = stat.iloc[idx]
            x_stat[['age']] = preprocess.zero_one_scale(x_stat[['age']])
            x_seqs = seqs[idx]
            X.append([x_stat.to_numpy(),x_seqs])

    #train model with train test sets
    X_real_tr,X_real_te,X_syn_tr,X_syn_te = X[0],X[1],X[2],X[3]
    Y_real_tr,Y_real_te,Y_syn_tr,Y_syn_te = Y[0],Y[1],Y[2],Y[3]
        
    print('training trajectory model...')
    #build model and predict
    real_model = utility.trajectory_RNN_simple(output_size=Y_real_tr.shape[1])
    syn_model = utility.trajectory_RNN_simple(output_size=Y_syn_tr.shape[1])
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
    filename = 'trajectory_pred_accuracy.txt'
    with open(os.path.join(result_path,filename),'w') as f:
        f.write('Real accuracy: ' + str(real_acc) + '\n')
        f.write('Synthetic accuracy: ' + str(syn_acc) + '\n')


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

    #trajectory_prediction(real_df,syn_df,result_path)



    
    

        




    
    
    
    
    
    



