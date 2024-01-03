import keras 
from keras import layers
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score
import pandas as pd
#build utility model
class trajectory_RNN_simple(keras.Model):
    def __init__(self,output_size):
        super().__init__()
        self.dense = layers.Dense(100,activation='relu')
        self.recurrent = layers.LSTM(100,activation='relu')
        self.concat = layers.Concatenate(axis=1)
        self.process_1 = layers.Dense(100,activation='relu')
        self.process_2 = layers.Dense(50,activation='relu')
        self.classify = layers.Dense(output_size,activation='softmax')

    def call(self, inputs):
        attributes,sequences = inputs
        attr = self.dense(attributes)
        long = self.recurrent(sequences)
        x = self.concat([attr,long])
        x = self.process_1(x)
        x = self.process_2(x)
        return self.classify(x)
    
def accuracy(real,pred):
    return accuracy_score(real,pred)

def auc(labels,pred_scores):
    return roc_auc_score(labels,pred_scores)

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

class mortality_RNN_simple(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(100,activation='relu')
        self.recurrent = layers.LSTM(100,activation='relu')
        self.concat = layers.Concatenate(axis=1)
        self.process_1 = layers.Dense(100,activation='relu')
        self.process_2 = layers.Dense(50,activation='relu')
        self.classify = layers.Dense(1,activation='sigmoid')

    def call(self, inputs):
        attributes,longitudinal = inputs
        attr = self.dense(attributes)
        long = self.recurrent(longitudinal)
        x = self.concat([attr,long])
        x = self.process_1(x)
        x = self.process_2(x)
        return self.classify(x)