import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

# Properties
batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# One hot encoder
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def createModel():
    '''
        Create a CNN model\n
    '''

    # Squential model (layer by layer)
    model = Sequential()

    # Add a convolutional layer,
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1),
                     padding='same'))  # input_shape: (w,h,layers)
    # Except the first layer, there is no need to provide the input shape due to their connection
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    # Pooling is a layer which is used to reduce the image size
    # It works just like a convolutional layer, but no computation, just max or averate of an area is gotten
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout means to randomly set some attributes to zeros
    # Used to reduce bias
    model.add(Dropout(0.25))

    # A flatten layer converts an n-d array into a single dimension vector
    model.add(Flatten())

    # Dense is a normal layer just like in MLP
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # The output
    model.add(Dense(num_classes, activation='softmax'))

    # The model should be compiled before training
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


model = createModel()

# Fit the model to dataset
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
