from matplotlib import pyplot as plt
import numpy as np

class Fidelity:
    #class with reporting utilities for evaluating fidelity of synthetic data to real data
    def stats_table(descr_stats,statistics,features):
        #create a table of descriptive statistics, from input statistics of interest (mean, st dev, range, etc.) for all features
        #statistics are the row labels, features are the column labels

        #create numpy array of descriptive statistics
        table = np.empty((len(statistics),len(features)))
        j=0
        for i in statistics:
            table[j,:] = descr_stats[i]
            j+=1

        #create the table plot
        fig,ax=plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(table,loc='center',colLabels=features,rowLabels=statistics)
        fig.tight_layout()
        return plt


    def percentile_plot(long_stats,percentiles,alphas):
        #create longitudinal percentiles plot from longitudinal descriptive statistics
        #takes as input the percentiles we would like to show, and corresponding alphas (plot transparency)
        #TODO:
        #create subplot for each feature
        
        plt.figure('percentiles')
        for i in range(len(percentiles)):
            plt.plot(long_stats[f'percentile_{percentiles[i]}'][:,0],c='r',alpha=alphas[i])
        return plt

    def tsne_plot(tsne_embeddings):
        #plot the 'fitted' tsne 

        plt.figure('TSNE of longitudinal data')
        plt.scatter(tsne_embeddings[:,0],tsne_embeddings[:,1])
        return plt

    def gof_report(dct):
        #create a report table of the gof test and metric 
        return 'test_stat: ' +  str(dct['test_stat']) , ' p_value: ' + str(dct['p_val'])