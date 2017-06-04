from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.utils import plot_model
import keras
import pydot
import graphviz
import argparse
import time

class timeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_times = []
        self.epoch_times = []

    def on_batch_begin(self, batch, logs={}):
        self.batch_begin = time.time()

    def on_batch_end(self, batch, logs={}):
        self.batch_end = time.time()
        self.batch_times.append(self.batch_end-self.batch_begin)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_begin = time.time()
        self.batch_times = []

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_end = time.time()
        self.epoch_times.append(self.epoch_end-self.epoch_begin)
        print('End of epoch {}:{}'.format(
            len(self.epoch_times),
            self.epoch_end-self.epoch_begin)
              )
        print(self.batch_times)

parser = argparse.ArgumentParser(description='Experiment specs')
parser.add_argument('--run_date', type=str, help='run date')
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--architecture', type=str, help='model architecture')
parser.add_argument('--instance_type', type=str, help='instance type')

args = parser.parse_args()

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

plot_model(model, to_file='logs/{}_{}.png'.format(args.dataset, args.architecture))

csv_logger = keras.callbacks.CSVLogger('logs/{}_{}_{}_{}.out'.format(args.run_date, args.dataset, args.architecture, args.instance_type))
time_history = timeHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=[csv_logger, time_history])
