import csv
import os
from timeit import default_timer as timer

start = timer()

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

inputs = []
outputs = []

num_inputs = 784
num_outputs = 10
num_hidden = 100
num_epochs = 1

with open('../../data/mnist_train_inputs.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        prep = []
        for i in range (0, 784):
            prep.append(float(row[i]))
        inputs.append(prep)

with open('../../data/mnist_train_outputs.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        prep = []
        for i in range (0, 10):
            prep.append(float(row[i]))
        outputs.append(prep)


print("Time to load: {0:.20f}s".format(timer() - start)) 

start = timer()

model = Sequential()
model.add(Dense(num_hidden, input_dim=num_inputs))
model.add(Activation('sigmoid'))
model.add(Dense(num_outputs))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd)

print("Time to construct: {0:.20f}s".format(timer() - start))

start = timer()

model.fit(np.array(inputs), np.array(outputs), batch_size=1, epochs=num_epochs, verbose=0)

print("Time to train: {0:.20f}s".format(timer() - start))
start = timer()

_ = model.predict_proba(inputs, verbose=0)

print("Time to eval: {0:.20f}s".format(timer() - start))
