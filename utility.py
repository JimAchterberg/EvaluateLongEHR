import os 
import numpy as np
import pandas as pd

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


   
    from utils import preprocess
    X = []
    Y = []
    for i in [real_df,syn_df]:
        #sequence and target preprocessing
        seq = []
        y = []
        for sequence in preprocess.get_sequences(i,column='icd_code'):
            for t in range(len(sequence)-1):
                seq.append(sequence[:t+1])
                y.append(sequence[t+1])
        y = pd.DataFrame(y,columns=['target'])
        y = preprocess.one_hot_encoding(y,columns=['target'],column_sizes=[119])
        Y.append(y)
        max_sequence_length = max(len(s) for s in seq)
        seq = preprocess.sequences_to_3d(seq,maxlen=max_sequence_length,padding=-1)
        seq = preprocess.one_hot_3d(seq,119+1)
        seq = seq[:,:,1:]
        seq = seq.astype(float)
        #static data preprocessing
        static = preprocess.get_static(i,columns=['age','gender','deceased','race']) 
        static = preprocess.one_hot_encoding(static,columns=['race'],column_sizes=[6])
        static[['age']] = preprocess.zero_one_scale(static[['age']])
        #appending data to target list
        x = [static,seq] 
        X.append(x)


    X_real,X_syn = X[0],X[1]
    #TODO: train test split!!!
   

    




    
    
    
    
    
    



