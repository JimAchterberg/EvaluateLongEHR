import keras 
from keras import layers 
import numpy as np 

from matplotlib import pyplot as plt

class privacy_RNN(keras.Model):
    def __init__(self,labels,nodes_at_input=100):
        super().__init__()
        self.labels=labels
        self.recurrent_input = layers.LSTM(nodes_at_input,activation='relu')
        self.dense_input = layers.Dense(nodes_at_input,activation='relu')
        self.concat = layers.Concatenate(axis=1)
        self.process1 = layers.Dense(int(nodes_at_input),activation='relu')
        self.process2 = layers.Dense(int(nodes_at_input/2),activation='relu')
        self.output_age = layers.Dense(1,activation='linear',name='output_1')
        self.output_gender = layers.Dense(1,activation='sigmoid',name='output_2')
        self.race_size = sum(l.count('race') for l in self.labels)
        self.output_race = layers.Dense(self.race_size,activation='softmax',name='output_3')
        
        
    def call(self, inputs):
        attr, long = inputs 
        attr = self.dense_input(attr)
        long = self.recurrent_input(long)
        x = self.concat([attr, long])
        x = self.process1(x)
        x = self.process2(x)
        #specify which outputs are used
        outputs = []
        if 'age' in self.labels:
            outputs.append(self.output_age(x))
        if 'gender' in self.labels:
            outputs.append(self.output_gender(x))
        if self.race_size>0:
            outputs.append(self.output_race(x))
        return outputs
    
#if __name__=='__main__':
    # labels = ['age','gender','race_0','race_1','race_2']
    # race_size = sum(l.count('race') for l in labels)
    # model = privacy_RNN(labels)

    # xlong = np.random.normal(0,1,(100,10,5))
    # xattr = np.random.normal(0,1,(100,3))
    # yage = np.random.normal(50,20,(100,1))
    # ygender = np.random.randint(0,2,(100,1))
    # yrace = np.random.randint(0,2,(100,race_size))

    # #custom losses and metrics
    # losses = {}
    # metrics = {}
    # if 'age' in labels:
    #     losses['output_1'] = 'mse'
    #     metrics['output_1'] = 'mse'
    # if 'gender' in labels:
    #     losses['output_2'] = 'binary_crossentropy'
    #     metrics['output_2'] = 'accuracy'
    # if race_size>0:
    #     losses['output_3'] = 'categorical_crossentropy'
    #     metrics['output_3'] = 'accuracy'


    # model.compile(optimizer='Adam',loss=losses,metrics=metrics)
    # fitted_model = model.fit([xattr,xlong],[yage,ygender,yrace],epochs=100,validation_split=.2)
    # preds = model.predict([xattr,xlong])



    # #plot the metrics (and save?)
    # accs = [x for x in fitted_model.history.keys() if 'acc' in x]
    # lss = [x for x in fitted_model.history.keys() if 'loss' in x]
    # mses = [x for x in fitted_model.history.keys() if 'mse' in x]


    # for key in accs:
    #     plt.plot(fitted_model.history[key])
    # plt.legend(accs)
    # plt.show()


