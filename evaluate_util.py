import numpy as np
from sklearn.manifold import TSNE
from dtwParallel import dtw_functions
import keras 
from keras import layers
from scipy.stats import ks_2samp



class Fidelity:

    def create_dicts(attr,long,syn_attr,syn_long):
        #create dictionary of all possible descriptive statistics (in the future we can use this to only select particular statistics for our table)
            attr_dct = {}
            syn_attr_dct = {}
            long_dct = {}
            syn_long_dct = {}
            #MEAN
            attr_dct['mean'] = np.mean(attr,axis=0)
            syn_attr_dct['mean'] = np.mean(syn_attr,axis=0)
            long_dct['mean'] = np.mean(long,axis=(0,1))
            syn_long_dct['mean'] = np.mean(syn_long,axis=(0,1))
            #STDEV
            attr_dct['st.dev'] = np.std(attr,axis=0)
            syn_attr_dct['st.dev'] = np.std(syn_attr,axis=0)
            long_dct['st.dev'] = np.std(long,axis=(0,1))
            syn_long_dct['st.dev'] = np.std(syn_long,axis=(0,1))
            #MIN
            attr_dct['min'] = np.min(attr,axis=0)
            syn_attr_dct['min'] = np.min(syn_attr,axis=0)
            long_dct['min'] = np.min(long,axis=(0,1))
            syn_long_dct['min'] = np.min(syn_long,axis=(0,1))
            #MAX
            attr_dct['max'] = np.max(attr,axis=0)
            syn_attr_dct['max'] = np.max(syn_attr,axis=0)
            long_dct['max'] = np.max(long,axis=(0,1))
            syn_long_dct['max'] = np.max(syn_long,axis=(0,1))
            return attr_dct,syn_attr_dct,long_dct,syn_long_dct
    
    def percentiles(long,syn_long,percentiles):
        #create dictionaries of percentiles over time for real and synthetic data
        perc_dct = {}
        syn_perc_dct = {}

        for i in percentiles:
            perc_dct[str(i)] = np.percentile(long,q=i,axis=0)
            syn_perc_dct[str(i)] = np.percentile(syn_long,q=i,axis=0)

        return perc_dct,syn_perc_dct



    def tsne(data):
        #computes tsne embeddings for longitudinal mixed-type data (using DTW with gower distance)
        
        #create a configuration object for the distance matrix algorithm
        class Input:
            def __init__(self):
                self.check_errors = False 
                self.type_dtw = "d"
                self.constrained_path_search = "sakoe_chiba"
                self.MTS = True
                self.regular_flag = False
                self.n_threads = -1
                self.local_dissimilarity = "gower"
                self.visualization = False
                self.output_file = False
                self.dtw_to_kernel = False
                self.sigma_kernel = 1
                self.itakura_max_slope = None
                self.sakoe_chiba_radius = 1
        input_obj = Input()
        #find distance matrix of data to itself
        dist_matrix = dtw_functions.dtw_tensor_3d(data,data,input_obj)
        #input to TSNE is distance matrix of our sample
        embeddings = TSNE(n_components=2,init='random',metric='precomputed').fit_transform(dist_matrix)
        return embeddings

    def gof(attr_real,long_real,attr_syn,long_syn):
        #performs goodness-of-fit test on synthetic data (longitudinal and attributes) using classification-based testing

        #data shape parameters for model specification
        K = attr_real.shape[1]
        T = long_real.shape[1]
        F = long_real.shape[2]

        #the gof model predicts binary labels on a dataset of real (0) and synthetic (1) data
        attr = np.concatenate((attr_real,attr_syn),axis=0)
        long = np.concatenate((long_real,long_syn),axis=0)
        y_real = np.zeros(shape=(attr_real.shape[0]))
        y_syn = np.ones(shape=(attr_syn.shape[0]))
        y = np.concatenate((y_real,y_syn),axis=0)
        N = y.shape[0]

        #shuffle the data (and corresponding labels) and make a 60% train/test split
        shuffle = np.random.permutation(N)
        meta,long,y = attr[shuffle],long[shuffle],y[shuffle]
        split_size = int(.6*N)
        attr_train,attr_test = attr[:split_size],attr[split_size:]
        long_train,long_test = long[:split_size],long[split_size:]
        y_train,y_test = y[:split_size],y[split_size:]

        #specifying the model
        #input layers:
        a_input = keras.Input(shape=(K,))
        l_input = keras.Input(shape=(T,F))
        #hidden layers (relu for nonlinear relations):
        a = layers.Dense(100,activation='relu')(a_input)
        l = layers.LSTM(100,activation='relu')(l_input)
        x = layers.Concatenate(axis=1)([a,l])
        x = layers.Dense(50,activation='relu')(x)
        #output layer (softmax for predicting binary label):
        output = layers.Dense(1,activation='sigmoid')(x)
        model = keras.Model(inputs=[a_input,l_input],outputs=output,name='gof_model')
        #compile with adam optimizer, binary crossentropy loss, and accuracy metric
        model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='accuracy')

        #train the model
        model.fit(x=[attr_train,long_train],y=y_train,batch_size=16,epochs=100,)

        #make predictions on the binary label (real 0 / synthetic 1)
        pred = model.predict(x=[attr_test,long_test])

        #run kolmogorov smirnoff test on predictions of real versus synthetic samples
        pred_real = pred[y_test==0]
        pred_syn = pred[y_test==1]
        test_stat,p_val = ks_2samp(data1=pred_real.flatten(),data2=pred_syn.flatten(),alternative='two-sided')

        #save outputs for report in a dictionary
        dct = {}
        dct['test_stat'] = test_stat
        dct['p_val'] = p_val
        dct['pred_real'] = pred_real
        dct['pred_syn'] = pred_syn

        return dct



class Utility:
    #class with methods for evaluating utility of synthetic data in real-world tasks
    def forecasting():
        pass


class Privacy:
    #class with methods for evaluating privacy preserving capabilities of synthetic data
    def aia():
        pass