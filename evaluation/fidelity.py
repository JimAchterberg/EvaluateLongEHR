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
    diff_matrixplot.title('Synthetic/real ICD section frequency difference')
    filename = 'freq_diff_matrixplot.png'
    diff_matrixplot.savefig(os.path.join(result_path,filename))
    diff_matrixplot.show()

#executes the tsne step
def exec_tsne(real_df,syn_df,result_path):
    df = pd.concat([real_df,syn_df],axis=0)
    static = preprocess.get_static(df,['age','gender','deceased','race']).astype(float)
    seq = preprocess.df_to_3d(df,'icd_code',padding=-1).astype(str)

    #find distance matrices
    static_distances = fidelity.static_gower_matrix(static,cat_features=[False,True,True,True])
    timevarying_distances = fidelity.mts_gower_matrix(seq)#,cat_features=[True])

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
    tsne_plot = fidelity.tsne(distance_matrix,labels)
    #tsne_plot.title('tSNE plot of synthetic/real samples')
    filename = 'tsne.png'
    tsne_plot.savefig(os.path.join(result_path,filename))
    tsne_plot.show()
    
#executes gof step
def exec_gof(real_df,syn_df,result_path):
    labels = np.concatenate((np.zeros(shape=(real_df.subject_id.nunique())),
                           np.ones(shape=(syn_df.subject_id.nunique()))),axis=0)
    
    #preprocessing for model
    df = pd.concat([real_df,syn_df],axis=0)
    #one hot encode before splitting data to ensure proper encoding
    for col in ['race','icd_code']:
        dummies = pd.get_dummies(df[col],prefix=col)
        df = df.drop(col,axis=1)
        df = pd.concat([df,dummies],axis=1)
    
    seq = preprocess.df_to_3d(df,cols=[x for x in df.columns if 'icd_code' in x],padding=0)
    static = preprocess.get_static(df,columns=[x for x in df.columns if 'icd_code' not in x])

    #pick 70% random indices stratified on synthetic/real labels
    train_idx,test_idx,y_train,y_test = preprocess.train_split(X=np.arange(seq.shape[0]),y=labels,stratify=labels)
    X = []
    for idx in [train_idx,test_idx]:
        #select train/test data
        x_seq = seq[idx]
        x_static = static.iloc[idx]
        #zero one scale numerical variables
        x_static[['age']] = preprocess.zero_one_scale(x_static[['age']])
        #append data to data list
        X.append([x_static.to_numpy().astype(float),x_seq.astype(float)])
    

    #fit a keras model and perform GoF test
    model = fidelity.gof_model()
    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='accuracy')
    model.fit(X[0],y_train,batch_size=32,epochs=1,validation_split=.2)
    pred = model.predict(X[1])
    test_stat,pval = fidelity.ks_test(real_pred=pred[y_test==0],syn_pred=pred[y_test==1])

    #additional numbers for final report
    accuracy = fidelity.accuracy(y_test,np.round(pred))
    total_real = y_test.shape[0]-sum(y_test)
    total_syn = sum(y_test)
    correct_real = np.sum((y_test.flatten()==np.round(pred).flatten())[y_test.flatten()==0])
    correct_syn = np.sum((y_test.flatten()==np.round(pred).flatten())[y_test.flatten()==1])

    #make a report of results
    filename = 'gof_test_report.txt'
    with open(os.path.join(result_path,filename),'w') as f:
        f.write('accuracy: ' + str(accuracy) + '\n')
        f.write('correct real: ' + str(correct_real) + '\n')
        f.write('correct synthetic: ' + str(correct_syn) + '\n')
        f.write('total real: ' + str(total_real) + '\n')
        f.write('total synthetic: ' + str(total_syn) + '\n')
        f.write('p-value: ' + str(pval) + '\n')

    
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
    sample_size = 100
    split = np.random.choice(real_df.subject_id.unique(),int(real_df.subject_id.nunique()/2))
    d = real_df.subject_id.unique()
    split1 = np.random.choice(d,sample_size)
    d = [x for x in d if x not in split1]
    split2 = np.random.choice(d,sample_size)
    syn_df = real_df[real_df.subject_id.isin(split1)]
    real_df = real_df[real_df.subject_id.isin(split2)]
   

    #execute the different steps (or comment out if you do not wish to perform a step)
    #exec_descr_stats(real_df,syn_df,result_path)
    #exec_tsne(real_df,syn_df,result_path)
    exec_gof(real_df,syn_df,result_path) 