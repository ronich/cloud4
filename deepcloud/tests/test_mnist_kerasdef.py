import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
import numpy as np
import pydot
import graphviz
import argparse

parser = argparse.ArgumentParser(description='Experiment specs')
parser.add_argument('--run_date', type=str, help='run date')
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--architecture', type=str, help='model architecture')

args = parser.parse_args()


# In[3]:

batch_size = 128
num_classes = 10
epochs = 12


# In[4]:

img_rows, img_cols = 28, 28


# In[5]:

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# In[7]:

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[8]:

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[9]:

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
metrics=['accuracy'])

# In[ ]:

plot_model(model, to_file='{}_{}.png'.format(args.dataset, args.architecture))

# In[32]:

csv_logger = keras.callbacks.CSVLogger('{}_{}_{}.out'.format(args.run_date, args.dataset, args.architecture))

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=[csv_logger])
