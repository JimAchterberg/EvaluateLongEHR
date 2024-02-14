import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#numerically encode categoricals, one hot if necessary, separate static and sequential data
def prepr(real_df,syn_df):
        n_real = real_df.subject_id.nunique()
        #pool samples for encoding
        df = pd.concat([real_df,syn_df],axis=0)
        #encode categoricals
        for col in ['gender','deceased','race','icd_code']:
            df[col],_ = pd.factorize(df[col])
        #one hot encode
        for col in ['race','icd_code']:
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)
            df = df.drop(col,axis=1)
        #separate static and sequential
        stat_columns = [x for x in df.columns if 'race' in x or x in ['age','gender','deceased']]
        dyn_columns = [x for x in df.columns if 'icd_code' in x]
        static = get_static(df,stat_columns)
        seq = df_to_3d(df,dyn_columns,padding=0)
        #unpool samples
        real = [static[:n_real],seq[:n_real]]
        syn = [static[n_real:],seq[n_real:]]
        return real,syn

#get static attribute table in which values are not repeated
def get_static(data,columns,subject_idx='subject_id'):
    return data.groupby(subject_idx)[columns].first()

#get 2d timevarying data to 3d numpy array (necessary when data is multi-column)
def df_to_3d(df,cols,subject_idx='subject_id',timestep_idx='seq_num',padding='-1',pad_to=None):
    #check if we pad to prespecified number of timesteps
    if pad_to == None:
        t = max(df[timestep_idx])
    else: 
        t = pad_to
    #check if we need an object np array or float
    if type(padding)==str:
        dtype=object
    else:
        dtype=int
    seq = np.full((df[subject_idx].nunique(),t,len(cols)),padding,dtype=dtype)
    for idx,(_,subject) in enumerate(df.groupby(subject_idx)[cols]):
        seq[idx,:subject.shape[0],:] = subject
    return seq



class Scaler():
    def __init__(self,method='zero-one'):
        super().__init__()
        self.method = method

    def transform(self,x):
        if self.method=='zero-one':
            self.min = x.min()
            self.max = x.max()
            return (x-self.min)/(self.max-self.min)
        
    def reverse_transform(self,x):
        if self.method=='zero-one':
            return x*(self.max-self.min) + self.min
        
    

def train_split(X,y,stratify=None,train_size=.7):
    #create train test split stratified on syn/real labels
    return train_test_split(X,y,stratify=stratify,train_size=train_size)



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
    stat = np.concatenate(stat)
    seqs = np.concatenate(seqs)
    x = [stat,seqs]
    y = np.concatenate(y)
    return x,y