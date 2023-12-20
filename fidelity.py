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
import utils


path = 'C:/Users/Jim/Documents/thesis_paper/data/mimic_iv_preprocessed'
file = 'PAR_real_data.csv'
cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
real_df = pd.read_csv(os.path.join(path,file),usecols=cols)

#REMOVE LATER!!!!!!!!!!
split = np.random.choice(real_df.subject_id.unique(),int(real_df.shape[0]/2))
syn_df = real_df[real_df.subject_id.isin(split)]
real_df = real_df[~real_df.subject_id.isin(split)]

#get static feature dataframes
real_df_static = utils.preprocess.get_static(data=real_df,columns=['age','gender','deceased','race'])
syn_df_static = utils.preprocess.get_static(data=syn_df,columns=['age','gender','deceased','race'])
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
#real_matrixplot = utils.fidelity.rel_freq_matrix_plot(data=real_df,columns=['icd_code'])
#syn_matrixplot = utils.fidelity.rel_freq_matrix_plot(data=syn_df,columns=['icd_code'])

# #------------------------------------------------------------------------------------------------------------
#tSNE step
#one-hot encode non-binary categoricals
# real_df = utils.preprocess.one_hot_encoding(real_df,columns=['race','icd_code'])
# syn_df = utils.preprocess.one_hot_encoding(syn_df,columns=['race','icd_code'])

# #get static feature dataframe with one hot encoded values
#static_cols = [x for x in real_df.columns if 'icd_code' not in x]
# real_df_static = utils.preprocess.get_static(data=real_df,columns=static_cols)
# syn_df_static = utils.preprocess.get_static(data=syn_df,columns=static_cols)

# separately find static and timevarying distances, first find static distances through gower package
#static_distances = utils.fidelity.static_gower_matrix(pd.concat([real_df_static,syn_df_static],axis=0))

# # # now get data to 3d array and find timevarying distances with dtw package
#timevarying_cols = [x for x in real_df.columns if 'icd_code' in x]
# real_3d = utils.preprocess.df_to_3d(data=real_df,timevarying_cols=timevarying_cols)
# syn_3d = utils.preprocess.df_to_3d(data=syn_df,timevarying_cols=timevarying_cols)
# timevarying_distances = utils.fidelity.mts_gower_matrix(data=pd.concat([real_3d,syn_3d],axis=0))

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