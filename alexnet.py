import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


num_classes = len(unpickle('data/batches.meta')[b'label_names'])
img_rows, img_cols, img_dims = 32, 32, 3


xtrain = None
ytrain = None
xtest = None
ytest = None

d = unpickle('data/test_batch')
xtest = d[b'data']
ytest = np.array(d[b'labels'])
xtest = np.reshape(xtest, newshape=(
    xtest.shape[0], img_rows, img_cols, img_dims), order='C')

for i in range(1, 6):
    d = unpickle('data/data_batch_' + str(i))
    x = d[b'data']
    y = np.array(d[b'labels'])
    if xtrain is None:
        xtrain = x
        ytrain = y
    else:
        xtrain = np.concatenate((xtrain, x), axis=0)
        ytrain = np.concatenate((ytrain, y))

xtrain = np.reshape(xtrain, newshape=(xtrain.shape[0], 32, 32, 3), order='C')
# Properties
batch_size = 16
epochs = 12

# One hot encoder
ytrain = keras.utils.to_categorical(ytrain, num_classes)
ytest = keras.utils.to_categorical(ytest, num_classes)

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)
datagen.fit(xtrain)


def alexnetModel():
    '''
        Create an Alexnet model model\n
    '''

    # Squential model (layer by layer)
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11, 11),
                     activation='relu',
                     input_shape=(img_rows, img_cols, img_dims),
                     strides=4))  # input_shape: (w,h,layers)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    # The output
    model.add(Dense(num_classes, activation='softmax'))

    # The model should be compiled before training
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


model = alexnetModel()

xtrain = xtrain.astype('float32')/255
xtest = xtest.astype('float32')/255
ytrain = ytrain.astype('float32')/255
ytest = ytest.astype('float32')/255

# Fit the model to dataset
model.fit(xtrain, ytrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xtest, ytest))

score = model.evaluate(xtest, ytest, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
