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

    # input data: static features + dynamic features up until t=t
    # output data: dynamic features at t=t+1
    # preprocess df : one hot encode categoricals, scale numericals
    
   
    from utils import preprocess
    seqs = []
    for i in [real_df,syn_df]:
        seq = preprocess.get_sequences(i,column='icd_code')
        input_sequences = []
        output = []
        for sequence in seq:
            for t in range(len(sequence)-1):
                input_sequences.append(sequence[:t+1])
                output.append(sequence[t+1])

        #get sequences to 3d numpy array and pad with -1
        max_sequence_length = max(len(seq) for seq in input_sequences)
        input_sequences = preprocess.sequences_to_3d(input_sequences, maxlen=max_sequence_length,padding=-1)
        input_sequences = preprocess.one_hot_3d(input_sequences,119+1)
        input_sequences = input_sequences[:,:,1:] #remove -1 category
        seqs.append(input_sequences)
    print(seqs)
    print(seqs[0].shape)

    




    
    
    
    
    
    



