import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

def loadCNN():
    CNN = Sequential()
    CNN.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(150,150,3), kernel_initializer='he_normal'))
    CNN.add(BatchNormalization())
    CNN.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'))
    CNN.add(BatchNormalization())

    CNN.add(MaxPooling2D(3,3))

    CNN.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(150,150,3), kernel_initializer='he_normal'))
    CNN.add(BatchNormalization())
    CNN.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'))
    CNN.add(BatchNormalization())

    CNN.add(MaxPooling2D(3,3))

    CNN.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'))
    CNN.add(BatchNormalization())
    CNN.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'))
    CNN.add(BatchNormalization())
    CNN.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'))
    CNN.add(BatchNormalization())

    CNN.add(MaxPooling2D(3,3))

    CNN.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'))
    CNN.add(BatchNormalization())
    CNN.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'))
    CNN.add(BatchNormalization())
    CNN.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'))
    CNN.add(BatchNormalization())

    CNN.add(MaxPooling2D(3,3))

    CNN.add(GlobalAveragePooling2D())
    CNN.add( Dense( 1817, activation = 'softmax', name = 'Softmax' ) )
    CNN.summary()

    return CNN