import numpy as np
import pandas as pd
import os
from utils import preprocess,fidelity

#executes the descriptive statistics step
def exec_descr_stats(real_df,syn_df,result_path):
    #get static feature dataframes
    real_df_static = preprocess.get_static(data=real_df,columns=['age','gender','deceased','race'])
    syn_df_static = preprocess.get_static(data=syn_df,columns=['age','gender','deceased','race'])

    # get descriptive statistics for static numerical variables
    real_stats = fidelity.descr_stats(data=real_df_static[['age']])
    syn_stats = fidelity.descr_stats(data=syn_df_static[['age']])
    filename = 'descr_stats_staticnumerical.csv'
    print(real_stats)
    print(syn_stats)
    pd.concat([real_stats,syn_stats],axis=1).to_csv(os.path.join(result_path,filename))

    # #get relative frequencies for static categorical variables
    real_rel_freq = fidelity.relative_freq(data=real_df_static[['gender','deceased','race']])
    syn_rel_freq = fidelity.relative_freq(data=syn_df_static[['gender','deceased','race']])
    filename = 'descr_stats_staticcategorical.csv'
    print(real_rel_freq)
    print(syn_rel_freq)
    pd.concat([real_rel_freq,syn_rel_freq],axis=1).to_csv(os.path.join(result_path,filename))

    # #get matrixplot of relative frequencies for timevarying categorical variables over time
    real_freqmatrix = fidelity.rel_freq_matrix(data=real_df,columns='icd_code')
    syn_freqmatrix = fidelity.rel_freq_matrix(data=syn_df,columns='icd_code')
    diff_freqmatrix = real_freqmatrix-syn_freqmatrix
    diff_matrixplot = fidelity.freq_matrix_plot(diff_freqmatrix,range=(-.01,.01))
    filename = 'freq_diff_matrixplot.png'
    diff_matrixplot.savefig(os.path.join(result_path,filename))
    diff_matrixplot.show()

#executes the tsne step
def exec_tsne(real_df,syn_df,result_path):
    #get static feature dataframes
    real_df_static = preprocess.get_static(data=real_df,columns=['age','gender','deceased','race'])
    syn_df_static = preprocess.get_static(data=syn_df,columns=['age','gender','deceased','race'])
    #separately find static and timevarying distances, first find static distances through gower package
    real_df_static = real_df_static.astype(float)
    syn_df_static = syn_df_static.astype(float)
    #age gender deceased race
    static_distances = fidelity.static_gower_matrix(pd.concat([real_df_static,syn_df_static],axis=0),cat_features=[False,True,True,True])

    # now get data to 3d array and find timevarying distances with dtw package
    real_3d = preprocess.df_to_3d(data=real_df,columns=['icd_code'])
    syn_3d = preprocess.df_to_3d(data=syn_df,columns=['icd_code'])

    #only necessary for small sample checking since max timesteps might not be the same: pad df so they are of same length
    pad_steps = max(real_3d.shape[1],syn_3d.shape[1]) - min(real_3d.shape[1],syn_3d.shape[1])
    if real_3d.shape[1]<syn_3d.shape[1]:
        pad = np.full((real_3d.shape[0],pad_steps,real_3d.shape[2]),-1)
        real_3d = np.concatenate((real_3d,pad),axis=1)
    else:
        pad = np.full((syn_3d.shape[0],pad_steps,syn_3d.shape[2]),-1)
        syn_3d = np.concatenate((syn_3d,pad),axis=1)

    #ensure gower package recognizes categorical variables. it only recognizes non-numerical dtypes as categoricals when not explicitly specified
    real_3d = real_3d.astype(str)
    syn_3d = syn_3d.astype(str)
    #only icd_code is present
    timevarying_distances = fidelity.mts_gower_matrix(data=np.concatenate((real_3d,syn_3d),axis=0))#,cat_features=[True])

    # scale static and timevarying distances to similar range (while diagonal remains zero)
    static_distances = np.apply_along_axis(preprocess.zero_one_scale,0,static_distances)
    timevarying_distances = np.apply_along_axis(preprocess.zero_one_scale,0,timevarying_distances)

    # # take weighted sum of static and timevarying distances
    distance_matrix = (len(real_df_static.columns)/len(real_df))*static_distances + \
        ((len(real_df)-len(real_df_static))/len(real_df))*timevarying_distances
    filename = 'distance_matrix.csv'
    pd.DataFrame(distance_matrix).to_csv(os.path.join(result_path,filename))

    # #compute and plot tsne projections with synthetic/real labels as colors
    labels = np.concatenate((np.zeros(shape=(real_df_static.shape[0])),
                           np.ones(shape=(syn_df_static.shape[0]))),axis=0)
    tsne_plot = fidelity.tsne(distance_matrix,labels)
    filename = 'tsne.png'
    tsne_plot.savefig(os.path.join(result_path,filename))
    tsne_plot.show()

    #FIX tSNE PLOT LAYOUT!!!!
    
#executes gof step
def exec_gof(real_df,syn_df,result_path):
    #FIX!!!!
    labels = np.concatenate((np.zeros(shape=(real_df_static.shape[0])),
                           np.ones(shape=(syn_df_static.shape[0]))),axis=0)
    #preprocessing for model
    df = pd.concat([real_df,syn_df],axis=0)
    df = preprocess.one_hot_encoding(data=df,columns=['race','icd_code'])
    sbj_train,sbj_test,y_train,y_test = preprocess.train_split(X=df.subject_id.unique(),y=labels,stratify=labels,train_size=.7)
    X_train,X_test = df[df.subject_id.isin(sbj_train)],df[df.subject_id.isin(sbj_test)]
    X_train = [preprocess.get_static(X_train,columns=[x for x in X_train.columns if 'race' in x or x in ['gender','age','deceased']]).to_numpy().astype(float),
               preprocess.df_to_3d(data=X_train,columns=[x for x in X_train.columns if 'icd_code' in x]).astype(float)]
    X_test = [preprocess.get_static(X_test,columns=[x for x in X_test.columns if 'race' in x or x in ['gender','age','deceased']]).to_numpy().astype(float),
               preprocess.df_to_3d(data=X_test,columns=[x for x in X_test.columns if 'icd_code' in x]).astype(float)]
    
    #scale numerical variables!!!

    #instantiate a Keras model
    model = fidelity.gof_model()
    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='accuracy')
    model.fit(X_train,y_train,batch_size=32,epochs=10,validation_split=.2)
    pred = model.predict(X_test)
    test_stat,pval = fidelity.ks_test(real_pred=pred[y_test==0],syn_pred=pred[y_test==1])

    #additional numbers for final report
    accuracy = fidelity.accuracy(y_test,np.round(pred))
    total_real = y_test.shape[0]-sum(y_test)
    total_syn = sum(y_test)
    correct_real = np.sum((y_test.flatten()==np.round(pred).flatten())[y_test.flatten()==0])
    correct_syn = np.sum((y_test.flatten()==np.round(pred).flatten())[y_test.flatten()==1])

    #make a report of results
    columns = ['Correct','False','Total']
    rows = ['Real','Synthetic','Total','Accuracy','p_value']
    entries = [[correct_real,total_real-correct_real,total_real],[correct_syn,total_syn-correct_syn,total_syn],\
                [correct_real+correct_syn,y_test.shape[0]-(correct_real+correct_syn),y_test.shape[0]],\
                [np.nan,np.nan,accuracy],[np.nan,np.nan,pval]]
    gof = pd.DataFrame(entries,columns=columns,index=rows)
    print(gof)
    filename = 'gof_test_report.csv'
    gof.to_csv(os.path.join(result_path,filename))


if __name__ == '__main__':
    #load real and synthetic data
    path = 'C:/Users/Jim/Documents/thesis_paper/data/mimic_iv_preprocessed'
    file = 'real_data_221223.csv'
    version = 'v0.0'
    cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
    real_df = pd.read_csv(os.path.join(path,file),usecols=cols)

    result_path = os.path.join('results',version)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #REMOVE LATER!!!!!!!!!! ONLY NECESSARY FOR TESTING
    np.random.seed(123)
    sample_size = 50
    #split = np.random.choice(real_df.subject_id.unique(),int(real_df.subject_id.nunique()/2))
    d = real_df.subject_id.unique()
    split1 = np.random.choice(d,sample_size)
    d = [x for x in d if x not in split1]
    split2 = np.random.choice(d,sample_size)
    syn_df = real_df[real_df.subject_id.isin(split1)]
    real_df = real_df[real_df.subject_id.isin(split2)]

    #execute the different steps (or comment out if you do not wish to perform a step)
    #exec_descr_stats(real_df,syn_df,result_path)
    exec_tsne(real_df,syn_df,result_path)
    #exec_gof(real_df,syn_df)