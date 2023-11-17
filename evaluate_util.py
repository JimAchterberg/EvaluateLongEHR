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

    def train_gof(inputs,y,model,batch_size=32,epochs=100,validation_split=.2):
            attr,long = inputs
            model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='accuracy')
            model.fit(x=[attr,long],y=y,batch_size=batch_size,epochs=epochs,validation_split=validation_split)
            #save the fitted model
            return model

    def predict_gof(inputs,model):
        attr,long = inputs
        preds = model.predict(x=[attr,long])
        return preds
    
    def gof_test(real_pred,syn_pred):
        test_stat,p_val = ks_2samp(data1=real_pred.flatten(),data2=syn_pred.flatten(),alternative='two-sided')
        return p_val
        



class Utility:
    #class with methods for evaluating utility of synthetic data in real-world tasks
    def forecasting():
        pass


class Privacy:
    #class with methods for evaluating privacy preserving capabilities of synthetic data
    def aia():
        pass