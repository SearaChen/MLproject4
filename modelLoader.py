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
import sys
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

class ToDenseSeq(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        global encoder
        print("{} of 95".format(idx))
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return encoder.transform(batch_x),np.array(batch_y)

    def on_epoch_end(self):
        pass

if __name__ == '__main__':
    start = time.time()
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


    ag_train_data = read_in(path,'train')
    ag_test_data = read_in(path,'test')

    encoder = my_encoding()

    ag_train_labels = ag_train_data[:,0]
    #concatenate title and description

    if arg == 'yelp_pol':
        ag_train_text = ag_train_data[:,1]
    else:
        ag_train_text = [' '.join(s) for s in zip(ag_train_data[:,1], ag_train_data[:,2])]
    #print(np.shape(ag_train_text))
    #ag_train_text = np.asarray(ag_train_text)

    print("encoding...")
    encoder.fit(ag_train_text)

    ag_test_labels = ag_test_data[:,0]
    #concatenate title and description
    if arg == 'yelp_pol':
        ag_test_text = ag_test_data[:,1]
    else:
        ag_test_text = [' '.join(s) for s in zip(ag_test_data[:,1], ag_test_data[:,2])]

    print(np.shape(ag_test_text))
    ag_test_text = np.asarray(ag_test_text)




######CNN######
words = 128
chars =256

num_classes = len(set(ag_train_labels))

relabel = [l-1 for l in ag_train_labels]
Y_train = to_categorical(relabel, num_classes) # One-hot encode the labels
relabel = [l-1 for l in ag_test_labels]
Y_test = to_categorical(relabel, num_classes) # One-hot encode the labels

model = load_model("model.hdf5")

print(model.summary())

seq = ToDenseSeq(ag_test_text,Y_test,400)
print(model.evaluate_generator(seq,steps=testbatches))

print ("Time spent: {}s".format(time.time() -start))
