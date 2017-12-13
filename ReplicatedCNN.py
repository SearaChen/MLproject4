from encode import my_encoding
from ag_read_in import read_in
import numpy as np
from keras.utils import to_categorical
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Concatenate
from keras.models import Model
from keras.utils import Sequence
import math
import time
import sys
from keras.callbacks import ModelCheckpoint

"""
Reads in the data set, configures the CNN, and trains it on the training data. Then evaluates on the test set.
start with arguments, ag_news, dbpedia or yelp_pol to select the wanted data set.
"""


class ToDenseSeq(Sequence):
    """used to create the batches for the CNN"""
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        global encoder

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return encoder.transform(batch_x),np.array(batch_y)

    def on_epoch_end(self):
        pass

if __name__ == '__main__':
    start = time.time()

    #evaluate command line arguments and set parameters accordingly
    arg = sys.argv[1]
    path = arg + '_csv'
    if arg == 'ag_news':
        batches = 3750
        testbatches = 19
    elif arg == 'dbpedia':
        batches = 17500
        testbatches = 150
    elif arg == 'yelp_pol':
        path = 'yelp_review_polarity_csv'
        batches = 17500
        testbatches = 95

    #read in train and test data
    ag_train_data = read_in(path,'train')
    ag_test_data = read_in(path,'test')

    #create encoder
    encoder = my_encoding()

    ag_train_labels = ag_train_data[:,0]

    #concatenate title and description for datasets with both
    if arg == 'yelp_pol':
        ag_train_text = ag_train_data[:,1]
    else:
        ag_train_text = [' '.join(s) for s in zip(ag_train_data[:,1], ag_train_data[:,2])]

    #fit the encoder on the train set
    print("encoding...")
    encoder.fit(ag_train_text)


    ag_test_labels = ag_test_data[:,0]
    #concatenate title and description for datasets with both (test set
    if arg == 'yelp_pol':
        ag_test_text = ag_test_data[:,1]
    else:
        ag_test_text = [' '.join(s) for s in zip(ag_test_data[:,1], ag_test_data[:,2])]

    print(np.shape(ag_test_text))
    ag_test_text = np.asarray(ag_test_text)


######CNN######
#as specified in the original paper
words = 128
chars =256

#set the number of classes according to the data set
num_classes = len(set(ag_train_labels))

relabel = [l-1 for l in ag_train_labels]
Y_train = to_categorical(relabel, num_classes) # One-hot encode the labels
relabel = [l-1 for l in ag_test_labels]
Y_test = to_categorical(relabel, num_classes) # One-hot encode the labels

#network structure as provided in the paper
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

model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer='adam',                 # using the Adam optimiser
              metrics=['accuracy'])

print(model.summary())

#create individual batches with a Sequence
seq = ToDenseSeq(ag_train_text,Y_train,32)
#save the model after each epoch.
cb = ModelCheckpoint("model.hdf5", monitor='acc', save_best_only=False, save_weights_only=False, mode='auto', period=1)
#fit the model
model.fit_generator(seq,steps_per_epoch=batches, epochs=5, verbose=1,callbacks=[cb])

#after model is trained evaluate the test set
seq = ToDenseSeq(ag_test_text,Y_test,400)
print(model.evaluate_generator(seq,steps=testbatches))

#print total running time in seconds
print ("Time spent: {}s".format(time.time() -start))
