import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
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

# In[2]:

batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True


# In[15]:

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#a = np.random.randint(0, 50000, 10000)
#b = np.random.randint(0, 10000, 2000)
#(x_train, y_train), (x_test, y_test) = (x_train[a], y_train[a]), (x_test[b], y_test[b])


# In[16]:

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[17]:

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[18]:

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


# In[19]:

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
metrics=['accuracy'])


# In[20]:

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# In[ ]:

plot_model(model, to_file='{}_{}.png'.format(args.dataset, args.architecture))

# In[32]:

csv_logger = keras.callbacks.CSVLogger('{}_{}_{}.out'.format(args.run_date, args.dataset, args.architecture))

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=2,
          callbacks=[csv_logger])


# In[ ]:
