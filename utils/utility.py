import keras 
from keras import layers
from sklearn.metrics import accuracy_score
#build utility model
class trajectory_RNN_simple(keras.Model):
    def __init__(self,output_size):
        super().__init__()
        self.dense = layers.Dense(100,activation='relu')
        self.recurrent = layers.LSTM(100,activation='relu')
        self.concat = layers.Concatenate(axis=1)
        self.process_1 = layers.Dense(100,activation='relu')
        self.process_2 = layers.Dense(50,activation='relu')
        self.classify = layers.Dense(output_size,activation='softmax')

    def call(self, inputs):
        attributes,sequences = inputs
        attr = self.dense(attributes)
        long = self.recurrent(sequences)
        x = self.concat([attr,long])
        x = self.process_1(x)
        x = self.process_2(x)
        return self.classify(x)
    
def accuracy(real,pred):
    return accuracy_score(real,pred)