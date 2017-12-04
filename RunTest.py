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
import time


def batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    print(samples_per_epoch,number_of_batches)
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        denseList = []
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        denseList = [x.todense() for x in X_data[index_batch]]

        #X_batch = X_data[index_batch].todense()


        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(denseList),y_batch
        if (counter >= number_of_batches):
            counter=0




if __name__ == '__main__':
    start = time.time()

    encoder = my_encoding()
    ag_train_data = read_in('train')
    ag_test_data = read_in('test')

    #ag_train_data = np.asarray(ag_train_data)
    np.random.shuffle(ag_train_data)


    ag_train_labels = ag_train_data[:,0]
    #concatenate title and description
    ag_train_text = [' '.join(s) for s in zip(ag_train_data[:,1], ag_train_data[:,2])]
    print(np.shape(ag_train_text))
    ag_train_text = np.asarray(ag_train_text)

    print("encoding...")
    ag_train_input = encoder.fit(ag_train_text)
    #size,words,chars = np.shape(ag_train_input)
    print(np.shape(ag_train_input))
    ag_train_input = np.asarray(ag_train_input)

    """
    ag_test_labels = ag_test_data[:,0]
    #concatenate title and description
    ag_test_text = [' '.join(s) for s in zip(ag_test_data[:,1], ag_test_data[:,2])]
    print(np.shape(ag_test_text))
    ag_test_text = np.asarray(ag_test_text)

    ag_test_input = encoder.fit(ag_test_text)
    size,words,chars = np.shape(ag_test_input)
    print(np.shape(ag_test_input))
    ag_test_input = np.asarray(ag_test_input)
    """


######CNN######
words = 128
chars =256

num_classes = 4

relabel = [l-1 for l in ag_train_labels]
Y_train = to_categorical(relabel, num_classes) # One-hot encode the labels
#relabel = [l-1 for l in ag_test_labels]
#Y_test = to_categorical(relabel, num_classes) # One-hot encode the labels

input_1 = Input(shape=(words,chars))

conv1d_1 = Conv1D(256,3) (input_1)
conv1d_2 = Conv1D(256,3)(input_1)
conv1d_3 = Conv1D(256,3)(input_1)
conv1d_4 = Conv1D(256,3)(input_1)

max_pooling1d_1 = MaxPooling1D(3)(conv1d_1)
max_pooling1d_2 = MaxPooling1D(3)(conv1d_2)
max_pooling1d_3 = MaxPooling1D(3)(conv1d_3)
max_pooling1d_4 = MaxPooling1D(3)(conv1d_4)

merge_1 = Concatenate(axis=1)([max_pooling1d_1,max_pooling1d_2,max_pooling1d_3,max_pooling1d_4])

conv1d_5 = Conv1D(256,5)(merge_1)
max_pooling1d_5 = MaxPooling1D(3)(conv1d_5)

conv1d_6 = Conv1D(256,5)(max_pooling1d_5)
max_pooling1d_6 = MaxPooling1D(4)(conv1d_6)

flatten_1 = Flatten()(max_pooling1d_6)
dense_1 = Dense(128)(flatten_1)
dense_2 = Dense(num_classes)(dense_1)

model = Model(inputs=input_1, outputs=dense_2)

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

print(model.summary())

model.fit_generator(batch_generator(ag_train_input,Y_train,32), steps_per_epoch=3750, epochs=10,
                    verbose=1,workers=2,use_multiprocessing=True)

#model.fit(ag_train_input, Y_train,                # Train the model using the training set...
#          batch_size=32, epochs=5,
#          verbose=1,shuffle=True) #
#print(model.evaluate(ag_test_input,Y_test))

print ("Time spent: {}s".format(time.time() -start))
