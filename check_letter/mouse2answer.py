import matplotlib.pyplot as plt
import sklearn as sk
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import scipy
import sympy
import datetime
import os
from PIL import Image,ImageDraw
import time
import cv2
import sys

import check_letter


def to_dataset():
    path = 'D:/testset_one_char/19968/'
    oldx = oldy = -1 # 좌표 기본값 설정
    def on_mouse(event, x, y, flags, param):
        time = datetime.datetime.now()
        

        global oldx, oldy # 밖에 있는 oldx, oldy 불러옴

        if event == cv2.EVENT_LBUTTONDOWN: 
            oldx, oldy = x, y 

        elif event == cv2.EVENT_LBUTTONUP: # 마우스 뗐을때 발생
            #cv2.destroyAllWindows()
            cv2.imwrite(path+'answer.png',img)
            check_letter.print_answer()
        elif event == cv2.EVENT_MOUSEMOVE: # 마우스가 움직일 때 발생
            if flags & cv2.EVENT_FLAG_LBUTTON: 
                cv2.line(img, (oldx, oldy), (x, y), (0, 0, 0), 10, cv2.LINE_AA)
                cv2.imshow('image', img)
                oldx, oldy = x, y 

    # 흰색 컬러 영상 생성
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # 윈도우 창
    cv2.namedWindow('image')

    # 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
    # 마우스 이벤트가 발생하면 on_mouse 함수 실행
    cv2.setMouseCallback('image', on_mouse, img)

    # 영상 출력
    cv2.imshow('image', img)
    cv2.waitKey()
    
    
# 실행  
# to_dataset()