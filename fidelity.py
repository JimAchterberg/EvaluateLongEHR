#script performing evaluations for synthetic datasets
#from preprocess import data_loader, preprocess_mock
#from report_util import Fidelity as rep_fid
#import keras 
#from keras import layers

#-------------------------------------------------------------------------------------------------------
#load real and synthetic data
import numpy as np
import pandas as pd
import os
from utils import preprocess,fidelity


path = 'C:/Users/Jim/Documents/thesis_paper/data/mimic_iv_preprocessed'
file = 'real_data_211223.csv'
cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
real_df = pd.read_csv(os.path.join(path,file),usecols=cols)

#REMOVE LATER!!!!!!!!!!
np.random.seed(123)
sample_size = 50
#split = np.random.choice(real_df.subject_id.unique(),int(real_df.subject_id.nunique()/2))
d = real_df.subject_id.unique()
split1 = np.random.choice(d,sample_size)
d = [x for x in d if x not in split1]
split2 = np.random.choice(d,sample_size)
syn_df = real_df[real_df.subject_id.isin(split1)]
real_df = real_df[real_df.subject_id.isin(split1)]


#get static feature dataframes
real_df_static = preprocess.get_static(data=real_df,columns=['age','gender','deceased','race'])
syn_df_static = preprocess.get_static(data=syn_df,columns=['age','gender','deceased','race'])
#-----------------------------------------------------------------------------------------------------------
#descriptive statistics step

#get descriptive statistics for static numerical variables
# real_stats = utils.fidelity.descr_stats(data=real_df_static['age'])
# syn_stats = utils.fidelity.descr_stats(data=syn_df_static['age'])
# print(real_stats)
# print(syn_stats)

# # #get relative frequencies for static categorical variables
# real_rel_freq = utils.fidelity.relative_freq(data=real_df_static[['gender','deceased','race']])
# syn_rel_freq = utils.fidelity.relative_freq(data=syn_df_static[['gender','deceased','race']])

# # #get matrixplot of relative frequencies for timevarying categorical variables over time
# real_freqmatrix = fidelity.rel_freq_matrix(data=real_df,columns='icd_code')
# syn_freqmatrix = fidelity.rel_freq_matrix(data=syn_df,columns='icd_code')
# diff_freqmatrix = real_freqmatrix-syn_freqmatrix
# diff_matrixplot = fidelity.freq_matrix_plot(diff_freqmatrix,range=(-.01,.01))
# diff_matrixplot.show()

# #------------------------------------------------------------------------------------------------------------
#tSNE step

#separately find static and timevarying distances, first find static distances through gower package
# real_df_static = real_df_static.astype(float)
# syn_df_static = syn_df_static.astype(float)
# age gender deceased race
# static_distances = fidelity.static_gower_matrix(pd.concat([real_df_static,syn_df_static],axis=0),cat_features=[False,True,True,True])
# print(static_distances)

# # now get data to 3d array and find timevarying distances with dtw package

real_3d = preprocess.df_to_3d(data=real_df,timevarying_cols=['icd_code'])
syn_3d = preprocess.df_to_3d(data=syn_df,timevarying_cols=['icd_code'])
real_3d = real_3d.astype(float)
syn_3d = syn_3d.astype(float)

#only icd_code is present
timevarying_distances = fidelity.mts_gower_matrix(data=np.concatenate((real_3d,syn_3d),axis=0),cat_features=[True])
print(timevarying_distances)


# # normalize static and timevarying distances
# static_distances = static_distances.apply(utils.preprocess.normalize,axis=0)
# timevarying_distances = timevarying_distances.apply(utils.preprocess.normalize,axis=0)

# # take weighted sum of static and timevarying distances
# distance_matrix = (1/len(static_cols))*static_distances + (1/len(timevarying_cols))*timevarying_distances

# #compute and plot tsne projections with synthetic/real labels as colors
# labels = np.concatenate((np.zeros(shape=(real_df_static.shape[0])),
#                           np.ones(shape=(syn_df_static.shape[0]))),axis=0)
# tsne_plot = utils.fidelity.tsne(distance_matrix,labels)
# tsne_plot.show()

#-----------------------------------------------------------------------------------------------------------
#GoF step
# df = pd.concat([real_df,syn_df],axis=0)
# model = utils.fidelity.gof_model()
# X_train,X_test,y_train,y_test = utils.preprocess.gof_train_split(df=df,labels=labels)
#fitted_model = model.fit(x=[(utils.preprocess.get_static(X_train[static_cols])).to_numpy(),
#                            utils.preprocess.df_to_3d(X_train,columns=timevarying_cols)],
#                            y=y_train,model=model,batch_size=32,epochs=100,validation_split=.2)

# pred = model.predict(x=[(utils.preprocess.get_static(X_test[static_cols])).to_numpy(),
#                         utils.preprocess.df_to_3d(X_test,columns=timevarying_cols)])
# test_stat,pval = utils.fidelity.gof_test(real_pred=pred[y_test==0],syn_pred=pred[y_test==1])

# #additional numbers for final report
# accuracy = utils.preprocess.accuracy(y_test,pred)
# total_real = y_test.shape[0]-sum(y_test)
# total_syn = sum(y_test)
# correct_real = np.sum((y_test.flatten()==pred.flatten())[y_test.flatten()==0])
# correct_syn = np.sum((y_test.flatten()==pred.flatten())[y_test.flatten()==1])

# #make a report of results
# columns = ['Correct','False','Total']
# rows = ['Real','Synthetic','Total','Accuracy','p_value']
# entries = [[correct_real,total_real-correct_real,total_real],[correct_syn,total_syn-correct_syn,total_syn],\
#             [correct_real+correct_syn,y_test.shape[0]-(correct_real+correct_syn),y_test.shape[0]],\
#             [np.nan,np.nan,accuracy],[np.nan,np.nan,pval]]
# df = pd.DataFrame(entries,columns=columns,index=rows)
# print(df)