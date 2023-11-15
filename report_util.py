from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from evaluate_util import Fidelity as eval_fid

class Fidelity:
    #class with reporting utilities for evaluating fidelity of synthetic data to real data
    def stats_table(real,syn,feature_names,stats):
        #create a table of descriptive statistics, from input statistics of interest (mean, st dev, range, etc.) for all features
        #statistics are the row labels, features are the column labels

        #TODO: 
        # INCORPORATE ROW NAMES OF REAL/SYN
        # MAKE TABLE AS MATPLOTLIB TABLE INSTEAD OF DATAFRAME

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
        #TODO:
        #create subtitles and legends

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
    

    def tsne_plot(tsne_embeddings):
        #plot the 'fitted' tsne 

        plt.figure('TSNE of longitudinal data')
        plt.scatter(tsne_embeddings[:,0],tsne_embeddings[:,1])
        return plt

    def gof_report(dct):
        #create a report table of the gof test and metric 
        return 'test_stat: ' +  str(dct['test_stat']) , ' p_value: ' + str(dct['p_val'])