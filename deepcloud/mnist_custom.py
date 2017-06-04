import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
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
        _d = dict()
        _d['dataset'] = args.dataset
        _d['architecture'] = args.architecture
        _d['instance_type'] = args.instance_type
        _d['epoch'] = len(self.epoch_times)
        _d['time'] = self.epoch_end-self.epoch_begin
        _d['batch_times'] = self.batch_times
        _d['loss'] = logs.get('loss')
        _d['val_loss'] = logs.get('val_loss')
        _d['acc'] = logs.get('acc')
        _d['val_acc'] = logs.get('val_acc')
        print(_d)

parser = argparse.ArgumentParser(description='Experiment specs')
parser.add_argument('--run_date', type=str, help='run date')
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--architecture', type=str, help='model architecture')
parser.add_argument('--instance_type', type=str, help='instance type')

args = parser.parse_args()

batch_size = 64
num_classes = 10
epochs = 36

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

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
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False
    )

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

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
    verbose=2,
    validation_data=(x_test, y_test),
    callbacks=[csv_logger,
               time_history])
