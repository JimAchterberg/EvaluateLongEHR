#contains all models and (distance) algorithms

import keras 
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from gower import gower_matrix
from dtwParallel import dtw_functions

#model for GoF testing
class GoF_RNN(keras.Model):
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

#model for patient trajectory forecasting
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
    

#models for mortality prediction
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
    
class mortality_LR(LogisticRegression):
    def __init__(self, penalty='elasticnet', l1_ratio=0.5,):
        super().__init__(
            penalty=penalty,
            solver='saga',  # 'saga' solver supports both 'l1' and 'elasticnet' penalties
            l1_ratio=l1_ratio,
            max_iter=1000
        )
        
class mortality_RF(RandomForestClassifier):
    def __init__(self, n_estimators=100,max_depth=None):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
    
#tSNE projection plot, color coded by label
def tsne(distance_matrix,labels):
    embeddings = TSNE(n_components=2,init='random',metric='precomputed').fit_transform(distance_matrix)
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings[:,0],embeddings[:,1],c=labels,cmap='bwr')
    handles=[plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='blue', label='Real'),plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='red', label='Synthetic')]
    plt.legend(handles=handles)
    return plt

#get gower matrix for 2d data
def static_gower_matrix(data,cat_features=None):
    return gower_matrix(data,cat_features=cat_features)

#get gower matrix for 3d time series of mixed datatypes
def mts_gower_matrix(data,cat_features=None):
    class Input:
        def __init__(self):
            self.check_errors = True 
            self.type_dtw = "d"
            self.constrained_path_search = None
            self.MTS = True
            self.regular_flag = '-1'
            self.n_threads = -1
            self.local_dissimilarity = "gower"
            self.visualization = False
            self.output_file = False
            self.dtw_to_kernel = False
            self.sigma_kernel = 1
            self.itakura_max_slope = None
            self.sakoe_chiba_radius = None
    input_obj = Input()
    #to see progress, we can import tqdm and use it in dtwParallel package -> @ dtw_functions.dtw_tensor_3d 
    timevarying_distance = dtw_functions.dtw_tensor_3d(data,data,input_obj,cat_features)
    return timevarying_distance

class privacy_RNN(keras.Model):
    def __init__(self,labels):
        super().__init__()
        self.labels=labels
        nodes_at_input = 100
        self.recurrent_input = layers.LSTM(nodes_at_input,activation='relu')
        self.dense_input = layers.Dense(nodes_at_input,activation='relu')
        self.concat = layers.Concatenate(axis=1)
        self.process1 = layers.Dense(int(nodes_at_input),activation='relu')
        self.process2 = layers.Dense(int(nodes_at_input/2),activation='relu')
        self.output_age = layers.Dense(1,activation='linear',name='output_1')
        self.output_gender = layers.Dense(1,activation='sigmoid',name='output_2')
        self.race_size = sum(l.count('race') for l in self.labels)
        self.output_race = layers.Dense(self.race_size,activation='softmax',name='output_3')
        
        
    def call(self, inputs):
        attr, long = inputs 
        attr = self.dense_input(attr)
        long = self.recurrent_input(long)
        x = self.concat([attr, long])
        x = self.process1(x)
        x = self.process2(x)
        #specify which outputs are used
        outputs = []
        if 'age' in self.labels:
            outputs.append(self.output_age(x))
        if 'gender' in self.labels:
            outputs.append(self.output_gender(x))
        if self.race_size>0:
            outputs.append(self.output_race(x))
        return outputs