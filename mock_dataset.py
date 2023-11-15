#script to generate a mock dataset, to test evaluate capabilities

import numpy as np
import os

#datasize
n,t,k,f = (100,10,5,5)
#generate some random 'real' and 'synthetic' data
long = np.random.normal(loc=0.0,scale=100.0,size=(n,t,f))
attr = np.random.normal(loc=20.0,scale=20.0,size=(n,k))
syn_long = np.random.normal(loc=2.0,scale=115.0,size=(n,t,f))
syn_attr = np.random.normal(loc=23.0,scale=25.0,size=(n,k))

#flatten longitudinal data to 2d to save in csv
long = np.reshape(long,(n*t,f))
syn_long = np.reshape(syn_long,(n*t,f))

#save as csv file in correct directory, while first creating directory and files if not exists
if not os.path.exists('data/'):
    os.mkdir('data/')
    
data_dirs = ['data/real','data/syn']
files = []
for i in data_dirs:
    #making data directories
    if not os.path.exists(i):
        os.mkdir(i)
    #adding data files to list
    for j in ['mock_attr.csv','mock_longitudinal.csv']:
        files.append(os.path.join(i,j))


for i,j in zip(files,[attr,long,syn_attr,syn_long]):
    f = open(i,'w+')
    f.close()
    np.savetxt(i,j,delimiter=',')