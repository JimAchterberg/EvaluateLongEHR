#executes descriptive statistics and tsne step
#note that GoF test is part of utility.py, since it uses preprocessed data!
#tSNE and descriptive statistics use the raw data...

import pandas as pd
import numpy as np 
import os
from utils import preprocess,metrics,models

#executes the descriptive statistics step
def exec_descr_stats(real_df,syn_df,result_path):
    #get static feature dataframes
    real_df_static = preprocess.get_static(data=real_df,columns=['age','gender','deceased','race'])
    syn_df_static = preprocess.get_static(data=syn_df,columns=['age','gender','deceased','race'])

    # get descriptive statistics for static numerical variables
    real_stats = metrics.descr_stats(data=real_df_static[['age']])
    syn_stats = metrics.descr_stats(data=syn_df_static[['age']])
    filename = 'descr_stats_staticnumerical.csv'
    pd.concat([real_stats,syn_stats],axis=1).to_csv(os.path.join(result_path,filename))

    # #get relative frequencies for static categorical variables
    real_rel_freq = metrics.relative_freq(data=real_df_static[['gender','deceased','race']])
    syn_rel_freq = metrics.relative_freq(data=syn_df_static[['gender','deceased','race']])
    filename = 'descr_stats_staticcategorical.csv'
    pd.concat([real_rel_freq,syn_rel_freq],axis=1).to_csv(os.path.join(result_path,filename))

    # #get matrixplot of relative frequencies for timevarying categorical variables over time
    real_freqmatrix = metrics.rel_freq_matrix(data=real_df,columns='icd_code')
    syn_freqmatrix = metrics.rel_freq_matrix(data=syn_df,columns='icd_code')
    diff_freqmatrix = real_freqmatrix-syn_freqmatrix
    diff_matrixplot = metrics.freq_matrix_plot(diff_freqmatrix,range=None)
    diff_matrixplot.title('Synthetic/real ICD section frequency difference')
    filename = 'freq_diff_matrixplot.png'
    diff_matrixplot.savefig(os.path.join(result_path,filename))
    #diff_matrixplot.show()

#executes the tsne step
def exec_tsne(real_df,syn_df,result_path):
    df = pd.concat([real_df,syn_df],axis=0)
    static = preprocess.get_static(df,['age','gender','deceased','race']).astype(float)
    seq = preprocess.df_to_3d(df,cols=['icd_code'],padding=-1).astype(str)

    #find distance matrices
    static_distances = models.static_gower_matrix(static,cat_features=[False,True,True,True])
    timevarying_distances = models.mts_gower_matrix(seq)#,cat_features=[True])

    # scale static and timevarying distances to similar range (while diagonal remains zero)
    static_distances = np.apply_along_axis(preprocess.zero_one_scale,0,static_distances)
    timevarying_distances = np.apply_along_axis(preprocess.zero_one_scale,0,timevarying_distances)

    # # take weighted sum of static and timevarying distances
    distance_matrix = ((len(static.columns))/len(df.columns))*static_distances + \
        (seq.shape[2]/len(df.columns))*timevarying_distances
    filename = 'distance_matrix.csv'
    pd.DataFrame(distance_matrix).to_csv(os.path.join(result_path,filename))

    # #compute and plot tsne projections with synthetic/real labels as colors
    labels = np.concatenate((np.zeros(real_df.subject_id.nunique()),
                           np.ones(syn_df.subject_id.nunique())),axis=0)
    tsne_plot = models.tsne(distance_matrix,labels)
    #tsne_plot.title('tSNE plot of synthetic/real samples')
    filename = 'tsne.png'
    tsne_plot.savefig(os.path.join(result_path,filename))
    #tsne_plot.show()

if __name__ == '__main__':
    #load real and synthetic data
    path = 'C:/Users/Jim/Documents/thesis_paper/data'
    version = 'v0.0'
    syn_model = 'cpar'

    load_path = path + '/processed' + '/generated' 
    cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
    real_df = pd.read_csv(load_path+'/real/real.csv.gz',sep=',',compression='gzip',usecols=cols)
    syn_df = pd.read_csv(load_path+f'/{syn_model}/{syn_model}_{version}.csv.gz',sep=',',compression='gzip',usecols=cols)

    result_path = os.path.join('results',syn_model,version)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #exec_tsne(real_df,syn_df,result_path)
    exec_descr_stats(real_df,syn_df,result_path)