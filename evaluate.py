#script performing evaluations for synthetic datasets
#from preprocess import data_loader, preprocess_mock
#from report_util import Fidelity as rep_fid
#import keras 
#from keras import layers

#-------------------------------------------------------------------------------------------------------
#load real and synthetic data
import pandas as pd
import os
import utils
path = 'C:/Users/Jim/Documents/thesis_paper/data/mimic_iv_preprocessed'
file = 'PAR_real_data.csv'
cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
real_df = pd.read_csv(os.path.join(path,file),usecols=cols)

#state datatypes for simple accessing later
static_numerical = ['age']
static_categorical = ['gender','deceased','race']
timevarying_numerical = []
timevarying_categorical = ['icd_code']

#get static feature dataframe
real_df_static = utils.get_static(data=real_df,columns=static_categorical+static_numerical)

#-----------------------------------------------------------------------------------------------------------
#descriptive statistics step

#get descriptive statistics for static numerical variables
stats = utils.descr_stats(data=real_df_static[static_numerical])

#get relative frequencies for static categorical variables
rel_freq = utils.relative_freq(data=real_df_static[static_categorical])

#get relative frequencies for timevarying categorical variables
rel_freq = utils.relative_freq(data=real_df[timevarying_categorical])

#get matrixplot of relative frequencies for timevarying categorical variables over time
matrixplot = utils.rel_freq_matrix_plot(data=real_df,columns=timevarying_categorical,timestep_idx='seq_num')

#------------------------------------------------------------------------------------------------------------
#tSNE step

#one-hot encode non-binary categoricals
real_df = utils.one_hot_encoding(real_df,['race','icd_code'])
#get static feature dataframe with one hot encoded values
static_cols = [x for x in ]
real_df_static = utils.get_static(data=real_df,columns=)

#separately find static and timevarying distances, first find static distances through gower package
static_distances = utils.gower_matrix(real_df_static)

#now find timevarying distances, and get data to 3d numpy array
timevarying_cols = [x for x in real_df.columns if 'icd' in x]
real_3d = utils.df_to_3d(data=real_df,timevarying_cols=timevarying_cols)
timevarying_distances = utils.mts_gower_matrix(data=real_3d)

#normalize static and timevarying distances
static_distances = static_distances.apply(lambda x:(x-x.mean())/(x.std()),axis=0)
timevarying_distances = timevarying_distances.apply(lambda x:(x-x.mean())/(x.std()),axis=0)


#take sum of static and timevarying distances WEIGHTED BY the relative amount of static and timevarying distances


# #input to TSNE is distance matrix of our sample
# embeddings = TSNE(n_components=2,init='random',metric='precomputed').fit_transform(dist_matrix)
   
#-----------------------------------------------------------------------------------------------------------
#GoF step

#specify the GOF model yourself, for the data at hand
# class gof_model(keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.dense = layers.Dense(100,activation='relu')
#         self.recurrent = layers.LSTM(100,activation='relu')
#         self.concat = layers.Concatenate(axis=1)
#         self.process_1 = layers.Dense(100,activation='relu')
#         self.process_2 = layers.Dense(50,activation='relu')
#         self.classify = layers.Dense(1,activation='sigmoid')

#     def call(self, inputs):
#         attributes,longitudinal = inputs
#         attr = self.dense(attributes)
#         long = self.recurrent(longitudinal)
#         x = self.concat([attr,long])
#         x = self.process_1(x)
#         x = self.process_2(x)
#         return self.classify(x)
    
# report = rep_fid.gof_report(real_inputs=[attr,long],syn_inputs=[syn_attr,syn_long],model=gof_model())
# print(report)

