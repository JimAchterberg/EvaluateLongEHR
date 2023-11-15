import numpy as np
def data_loader(file_path):
    #loads data from a csv file as numpy array
    data = np.loadtxt(file_path,delimiter=',')
    return data

def preprocess_mock(attr,long):
        #preprocesses mock dataset
        n = attr.shape[0]
        t = int(long.shape[0]/n)
        f = long.shape[1]
        long = np.reshape(long,newshape=(n,t,f))
        return attr,long