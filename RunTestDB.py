from encode import my_encoding
from ag_read_in import read_in
import numpy as np
from keras.utils import to_categorical
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Concatenate
from keras.models import Model
from keras.utils import Sequence
import math
import time


class ToDenseSeq(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([x.todense() for x in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        pass

if __name__ == '__main__':
    start = time.time()

    encoder = my_encoding()
    ag_train_data = read_in('dbpedia_csv','train')
    ag_test_data = read_in('dbpedia_csv','test')

    ag_train_labels = ag_train_data[:,0]
    #concatenate title and description
    ag_train_text = [' '.join(s) for s in zip(ag_train_data[:,1], ag_train_data[:,2])]
    print(np.shape(ag_train_text))
    ag_train_text = np.asarray(ag_train_text)

    print("encoding...")
    ag_train_input = encoder.fit(ag_train_text)
    #size,words,chars = np.shape(ag_train_input)

    ag_train_input = np.asarray(ag_train_input)
    print(np.shape(ag_train_input))

    ag_test_labels = ag_test_data[:,0]
    #concatenate title and description
    ag_test_text = [' '.join(s) for s in zip(ag_test_data[:,1], ag_test_data[:,2])]
    print(np.shape(ag_test_text))
    ag_test_text = np.asarray(ag_test_text)

    ag_test_input = encoder.fit(ag_test_text)
    ag_test_input = np.asarray(ag_test_input)



######CNN######
words = 128
chars =256

num_classes = 14

relabel = [l-1 for l in ag_train_labels]
Y_train = to_categorical(relabel, num_classes) # One-hot encode the labels
relabel = [l-1 for l in ag_test_labels]
Y_test = to_categorical(relabel, num_classes) # One-hot encode the labels

input_1 = Input(shape=(words,chars))

conv1d_1 = Conv1D(256,3,activation='relu')(input_1)
conv1d_2 = Conv1D(256,3,activation='relu')(input_1)
conv1d_3 = Conv1D(256,3,activation='relu')(input_1)
conv1d_4 = Conv1D(256,3,activation='relu')(input_1)

max_pooling1d_1 = MaxPooling1D(3)(conv1d_1)
max_pooling1d_2 = MaxPooling1D(3)(conv1d_2)
max_pooling1d_3 = MaxPooling1D(3)(conv1d_3)
max_pooling1d_4 = MaxPooling1D(3)(conv1d_4)

merge_1 = Concatenate(axis=1)([max_pooling1d_1,max_pooling1d_2,max_pooling1d_3,max_pooling1d_4])

conv1d_5 = Conv1D(256,5,activation='relu')(merge_1)
max_pooling1d_5 = MaxPooling1D(3)(conv1d_5)

conv1d_6 = Conv1D(256,5,activation='relu')(max_pooling1d_5)
max_pooling1d_6 = MaxPooling1D(4)(conv1d_6)

flatten_1 = Flatten()(max_pooling1d_6)
dense_1 = Dense(128,activation='relu')(flatten_1)
dense_2 = Dense(num_classes, activation="softmax")(dense_1)

model = Model(inputs=input_1, outputs=dense_2)

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

print(model.summary())


seq = ToDenseSeq(ag_train_input,Y_train,32)

model.fit_generator(seq,steps_per_epoch=3750, epochs=5, verbose=1)

seq = ToDenseSeq(ag_test_input,Y_test,400)
print(model.evaluate_generator(seq,steps=19))

print ("Time spent: {}s".format(time.time() -start))
