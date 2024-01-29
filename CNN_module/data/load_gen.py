import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os 
from settings import ImageSize, Batch_Size

#__all__ = ["LoadGen"]

#class LoadGen:
'''
    def __init__(self):
        self.ImageSize = settings.ImageSize
        self.Batch_Size = settings.Batch_Size
'''

def gener(data_path):
    # 학습데이터 경로 설정
    RawDataPath = data_path

    # Keras 라이브러리의 ImageDataGenerator 클래스를 사용하여 데이터 생성기 만들기
    # 모델학습을 위해 이미지 데이터를 로드하고 이미지 증강
    DataGenerator = ImageDataGenerator(
        rescale=1./255,                 # 이미지 픽셀 0~1로 조정
        validation_split=0.2,           # 훈련데이터 0.8 / 검증데이터 0.2
        rotation_range=10,              # 이미지를 10도 범위 내에서 회전
        width_shift_range=0.1,          # (가로의 0.1 범위 내에서) 가로 방향으로 무작위 이동
        height_shift_range=0.1,         # (세로의 0.1 범위 내에서) 세로 방향으로 무작위 이동
        zoom_range=0.1,                 # 0.1배 무작위 확대/축소
        horizontal_flip=False           # 수평뒤집기 x
    )

    # 학습용 TrainGenerator
    # 상위 폴더 레이블링 및 전처리용 flow_from_directory 메서드
    TrainGenerator = DataGenerator.flow_from_directory(
        RawDataPath,
        target_size=ImageSize,
        batch_size=Batch_Size,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    # 검증용 ValGenerator
    ValGenerator = DataGenerator.flow_from_directory(
        RawDataPath,
        target_size=ImageSize,
        batch_size=Batch_Size,  
        class_mode='categorical',
        shuffle=True,
        subset='validation'
    )

    return TrainGenerator, ValGenerator