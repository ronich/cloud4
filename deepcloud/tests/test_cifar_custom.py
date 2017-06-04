import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import plot_model
import pydot
import graphviz
import argparse
import numpy as np
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
        print('Epoch {}:{}'.format(
            len(epoch_times),
            self.epoch_end-self.epoch_begin)
              )
        print(self.batch_times)

parser = argparse.ArgumentParser(description='Experiment specs')
parser.add_argument('--run_date', type=str, help='run date')
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--architecture', type=str, help='model architecture')
parser.add_argument('--instance_type', type=str, help='instance type')

args = parser.parse_args()

batch_size = 64
num_classes = 10
epochs = 300
data_augmentation = True

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
a = np.random.randint(0, 50000, 10000)
b = np.random.randint(0, 10000, 2000)
(x_train, y_train), (x_test, y_test) = (x_train[a], y_train[a]), (x_test[b], y_test[b])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# generating data
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False
    )

model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(768))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

plot_model(model, to_file='logs/{}_{}.png'.format(args.dataset, args.architecture))

csv_logger = keras.callbacks.CSVLogger('logs/{}_{}_{}_{}.out'.format(args.run_date, args.dataset, args.architecture, args.instance_type))
time_history = timeHistory()

datagen.fit(x_train)

model.fit_generator(
    datagen.flow(
        x_train, y_train, batch_size=batch_size
        ),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=2,
    callbacks=[csv_logger, time_history])
