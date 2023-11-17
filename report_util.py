from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from evaluate_util import Fidelity as eval_fid
from sklearn.metrics import accuracy_score

class Fidelity:
    #class with reporting utilities for evaluating fidelity of synthetic data to real data
    def stats_table(real,syn,feature_names,stats):
        #create a table of descriptive statistics, from input statistics of interest (mean, st dev, range, etc.) for all features
        #statistics are the row labels, features are the column label
        #separate data and extract dataset size
        attr,long = real
        syn_attr,syn_long = syn
        n,k = attr.shape
        _,t,f = long.shape
        attr_names,long_names = feature_names
        #create dictionaries with all possible descriptive statistics
        attr_dct,syn_attr_dct,long_dct,syn_long_dct = eval_fid.create_dicts(attr,long,syn_attr,syn_long)
        #create statistics table, which is the size of (2*#stats,long_features+attr_features)
        stats_table = np.empty(shape=(1,k+f))
        for i in stats:
            #concatenate attr and long statistic for syn and real
            attr_stat = np.expand_dims(attr_dct[i],0)
            long_stat = np.expand_dims(long_dct[i],0)
            real_stat = np.concatenate((attr_stat,long_stat),axis=1)
            syn_attr_stat = np.expand_dims(syn_attr_dct[i],0)
            syn_long_stat = np.expand_dims(syn_long_dct[i],0)
            syn_stat = np.concatenate((syn_attr_stat,syn_long_stat),axis=1)

            #append real and synthetic concatenated statistic to the stats table
            stats_table = np.append(stats_table,real_stat,axis=0)
            stats_table = np.append(stats_table,syn_stat,axis=0)
        stats_table = stats_table[1:]
        return pd.DataFrame(stats_table,columns=attr_names+long_names)
    



    def percentile_plot(long,syn_long,feature_names,percentiles,alphas):
        #creates percentile plots over time for longitudinal data to investigate stepwise distributions
        _,_,f = long.shape
        perc_dct,syn_perc_dct = eval_fid.percentiles(long,syn_long,percentiles)
        #output a plot of real and synthetic percentiles for the first feature
        fig,ax = plt.subplots(nrows=1,ncols=f)
        fig.suptitle('percentile plot')
        #we have f different plots
        for pl in range(f):
            #for which we plot each percentile
            j=0
            for i in percentiles:
                ax[pl].plot(perc_dct[str(i)][:,pl],c='b',alpha=alphas[j])
                ax[pl].plot(syn_perc_dct[str(i)][:,pl],c='r',alpha=alphas[j])
                ax[pl].set_title(feature_names[pl])
                j+=1
        return plt
    

    def tsne_plot(long,syn_long):
        n_r,t,f = long.shape
        n_s,_,_ = syn_long.shape
        #append synthetic to real data
        data = np.concatenate((long,syn_long),axis=0)
        #make binary labels for real(0) and synthetic(1) data to later colour the samples
        y = np.concatenate((np.zeros(shape=n_s),np.ones(shape=n_r)),axis=0)
        #find the 2d embeddings from tsne with gower & DTW
        embeddings = eval_fid.tsne(data)
        #plot the 'fitted' tsne 
        plt.figure('TSNE ')
        plt.scatter(embeddings[:,0],embeddings[:,1],c=y)
        return plt

    def gof_report(real_inputs,syn_inputs,model):
        #create a report table of the gof test and metric 
        real_attr,real_long = real_inputs
        syn_attr,syn_long = syn_inputs
        attr = np.concatenate((real_attr,syn_attr),axis=0)
        long = np.concatenate((real_long,syn_long),axis=0)
        n,t,f = long.shape
        _,k = attr.shape
        y_real = np.zeros(real_attr.shape[0])
        y_syn = np.ones(syn_attr.shape[0])
        y = np.concatenate((y_real,y_syn),axis=0)

        #shuffle data and create train test split
        shuffle = np.random.permutation(n)
        attr,long,y = attr[shuffle],long[shuffle],y[shuffle]
        split_size = int(.7*n)
        attr_train,attr_test = attr[:split_size],attr[split_size:]
        long_train,long_test = long[:split_size],long[split_size:]
        y_train,y_test = y[:split_size],y[split_size:]

        #fit the model and make predictions whether samples are real(0) or synthetic(1)
        fitted_model = eval_fid.train_gof(inputs=[attr_train,long_train],y=y_train,model=model)
        prob_preds = eval_fid.predict_gof(inputs=[attr_test,long_test],model=fitted_model)
        preds = np.round(prob_preds,decimals=0)

        #perform gof test
        real_preds = prob_preds[y_test==0]
        syn_preds = prob_preds[y_test==1]
        p_val = eval_fid.gof_test(real_preds,syn_preds)

        #additional numbers for final report
        acc = accuracy_score(y_test,preds)
        total_real = y_test.shape[0]-sum(y_test)
        correct_real = np.sum((y_test.flatten()==preds.flatten())[y_test.flatten()==0])
        total_syn = sum(y_test)
        correct_syn = np.sum((y_test.flatten()==preds.flatten())[y_test.flatten()==1])

        #make a report of results
        columns = ['Correct','False','Total']
        rows = ['Real','Synthetic','Total','Accuracy','p_value']

        entries = [[correct_real,total_real-correct_real,total_real],[correct_syn,total_syn-correct_syn,total_syn],\
                   [correct_real+correct_syn,y_test.shape[0]-(correct_real+correct_syn),y_test.shape[0]],\
                    [np.nan,np.nan,acc],[np.nan,np.nan,p_val]]
        
        df = pd.DataFrame(entries,columns=columns,index=rows)
        
        #           Correct     False   Total
        # Real      
        # Syn
        # Total                         N
        # Accuracy                      ...
        # p_value                       ...

        return df