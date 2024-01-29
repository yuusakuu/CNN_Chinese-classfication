import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from matplotlib.font_manager import findfont, FontProperties

from settings import Batch_Size, Epochs
from model import CNN_model

def run_model(TrainGenerator, ValGenerator, file_name, save_path): 
    checkpoint_callback = ModelCheckpoint(filepath=file_name, monitor='val_accuracy', save_best_only=True)

    # EarlyStopping: 에포크 중 val_accuracy이 5번 이상 변동이 없다면 학습중지
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    CNN = CNN_model.loadCNN()

    # Compile the model
    CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = CNN.fit(
        TrainGenerator,
        validation_data=ValGenerator,
        epochs=Epochs, 
        batch_size=Batch_Size,  
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # 모델 저장
    CNN.save(save_path)
    
    # 학습 정확도 및 검증 정확도 시각화
    Train_Accuracy = history.history['accuracy']
    Val_Accuracy = history.history['val_accuracy']
    Train_Loss = history.history['loss']
    Val_Loss = history.history['val_loss']
    epochs_range = range(Epochs)
    
    plt.figure( figsize=(14,4) )
    plt.subplot( 1,2,1 )
    plt.plot( range( len(Train_Accuracy) ), Train_Accuracy, label='Train' ) 
    plt.plot( range( len(Val_Accuracy) ), Val_Accuracy, label='Validation' ) 
    plt.legend( loc='lower right' )
    plt.title( 'Accuracy' )

    plt.subplot( 1,2,2 )
    plt.plot( range( len(Train_Loss) ), Train_Loss, label='Train' )
    plt.plot( range( len(Val_Loss) ), Val_Loss, label='Validation' )
    plt.legend( loc='upper right' )
    plt.title( 'Loss')

    plt.savefig('/mnt/d/download/test_result/savefig_default.png')
    return Train_Accuracy, Val_Accuracy, Train_Loss, Val_Loss, epochs_range
