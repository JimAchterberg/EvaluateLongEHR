#script to generate a mock dataset, to test generate and evaluate capabilities

import numpy as np
import os

#datasize
n,t,k,f = (100,10,5,5)
long = np.random.normal(loc=0.0,scale=100.0,size=(n,t,f))
attr = np.random.normal(loc=20.0,scale=20.0,size=(n,k))

#flatten longitudinal data to 2d to save in csv
long = np.reshape(long,(n*t,f))

#save as csv file in correct directory, while first creating directory and files if not exists
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

files = ['mock_longitudinal.csv','mock_attr.csv']
for i,j in zip(files,[long,attr]):
    f = open(os.path.join(data_dir,i),'w+')
    f.close()
    np.savetxt(os.path.join(data_dir,i),j,delimiter=',')