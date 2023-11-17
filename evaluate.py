#script performing evaluations for synthetic datasets
from preprocess import data_loader, preprocess_mock
from report_util import Fidelity as rep_fid
import keras 
from keras import layers

#-------------------------------------------------------------------------------------------------------

#load csv of mock dataset (or real & synthetic dataset in the future) and preprocess to 3d array
attr = data_loader('data/real/mock_attr.csv')
long = data_loader('data/real/mock_longitudinal.csv')
syn_attr = data_loader('data/syn/mock_attr.csv')
syn_long = data_loader('data/syn/mock_longitudinal.csv')
attr,long = preprocess_mock(attr,long)
syn_attr,syn_long = preprocess_mock(syn_attr,syn_long)
_,k = attr.shape
n,t,f = long.shape

attr_names = []
long_names = []
for i in range(k):
    attr_names.append(f'attribute_{i}')
for i in range(f):
    long_names.append(f'longitudinal_{i}')

#-------------------------------------------------------------------------------------------------------

#create descriptive statistics table
#TODO:
#ROWNAMES
# df = rep_fid.stats_table(real=[attr,long],syn=[syn_attr,syn_long],feature_names=[attr_names,long_names],stats=['mean','st.dev','min','max'])
# print(df)

#create percentile plot
# percentiles = [5,25,75,95]
# alphas = [.2,.5,.5,.2]
# perc_plot = rep_fid.percentile_plot(long,syn_long,long_names,percentiles,alphas)
# perc_plot.show()

#create tsne plot
# tsne_plot = rep_fid.tsne_plot(long,syn_long)
# tsne_plot.show()

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

