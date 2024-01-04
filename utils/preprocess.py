import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
#get static attribute table in which values are not repeated
def get_static(data,columns,subject_idx='subject_id'):
    return data.groupby(subject_idx)[columns].first()


#takes list of lists of sequences as input and transforms into padded 3d numpy array
def sequences_to_3d(list,maxlen,padding=-1):
    return np.expand_dims(pad_sequences(list, maxlen=maxlen, padding='post',value=padding),-1)

#get 2d timevarying data to 3d numpy array (necessary when data is multi-column)
def df_to_3d(df,cols,subject_idx='subject_id',timestep_idx='seq_num',padding=-1,pad_to=None):
    #check if we pad to prespecified number of timesteps
    if pad_to == None:
        t = max(df[timestep_idx])
    else: 
        t = pad_to
    seq = np.full((df[subject_idx].nunique(),t,len(cols)),padding)
    for idx,(_,subject) in enumerate(df.groupby(subject_idx)[cols]):
        seq[idx,:subject.shape[0],:] = subject
    return seq


#one hot encode non-binary categoricals
def one_hot_encoding(data,columns,column_sizes=None):
    for j,i in enumerate(columns):
        dummies = pd.get_dummies(data[i])
        dummies = dummies.reset_index(drop=True)
        if column_sizes[j]>dummies.shape[1]:
            zeros = pd.DataFrame(np.zeros(shape=(data.shape[0],column_sizes[j]-dummies.shape[1])))
            dummies = pd.concat([dummies,zeros],axis=1)
        column_names = []
        for n in range(dummies.shape[1]):
            column_names.append(str(i)+'_'+str(n))
        dummies.columns = column_names
        data = data.drop(i,axis=1)
        data = data.reset_index(drop=True)
        data = pd.concat([data,dummies],axis=1)
    return data

#normalizing function for pandas dataframe
def normalize(x):
    return (x-x.mean())/(x.std())

#zero one scaling function for pandas dataframe
def zero_one_scale(x):
    return (x-x.min())/(x.max()-x.min())
    

def train_split(X,y,stratify=None,train_size=.7):
    #create train test split stratified on syn/real labels
    return train_test_split(X,y,stratify=stratify,train_size=train_size)

#turns 2d dataframe into list of lists of variable length sequences
def get_sequences(df,column,subject_idx='subject_id',return_subject_idx=False):
    seq = []
    sbj = []
    for i in df[subject_idx].unique():
        seq.append(df[df[subject_idx]==i][column].to_list())
        sbj.append(i)
    if return_subject_idx:
        return seq,sbj
    else:
        return seq

#one hot encode 3d categorical array, while ensuring correct cardinality by concatting zeros
def one_hot_3d(data,cardinality):
    n,t,k = data.shape
    dummies = pd.get_dummies(data.flatten()) 
    if dummies.shape[1]<cardinality:
        dummies = dummies.reset_index()
        zeros = pd.DataFrame(np.zeros((dummies.shape[0],cardinality-dummies.shape[1])))
        dummies = pd.concat([dummies,zeros],axis=1)
    dummies = dummies.to_numpy().reshape((n,t,dummies.shape[1]))
    return dummies

def trajectory_input_output(x,max_t):
    static = x[0]
    seq = x[1]
    timesteps = np.max(np.where(np.any(seq!=0,axis=2),np.arange(seq.shape[1]),-1),axis=1)
    seqs = []
    stat = []
    y = []
    for t in range(1,max_t-1):
        #filter on rows which have max timesteps>=t+1
        x_seq = seq[timesteps>=t+1]
        x_stat = static[timesteps>=t+1]
        #input data is data up until t=t from seq (padded to end) and the corresponding static data
        stat.append(x_stat)
        x_seq = x_seq[:,:t,:]
        pad_size = max_t-x_seq.shape[1]
        x_seq = np.pad(x_seq,((0,0),(0,pad_size),(0,0)),'constant',constant_values=0)
        seqs.append(x_seq)
        #output data is seq at t=t+1
        y.append(seq[timesteps>=t+1][:,t+1,:])
    stat = pd.concat(stat,ignore_index=True)
    seqs = np.concatenate(seqs)
    x = [stat,seqs]
    y = np.concatenate(y)
    return x,y