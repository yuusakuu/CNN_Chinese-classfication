import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from matplotlib import pyplot as plt
import glob
from PIL import Image

def check_answer():
    # 모델이 있는 경로
    model_path = 'D:/fast_test_model/'
    # 제출한 답이 있는 경로
    test_path = 'D:/testset_one_char'

    # Load the trained model
    CNN = tf.keras.models.load_model(model_path+'CNN_Model.h5')
    # 테스트 실행
    Image_Size = (50, 50)
    Test_Data_Genetor = ImageDataGenerator( rescale=1./255 )
    Test_Generator = Test_Data_Genetor.flow_from_directory( test_path,target_size = Image_Size,
                                                            shuffle = False,class_mode = 'categorical' )
    # 예측한 결과 저장
    Test_Generator.reset()
    Predicts=CNN.predict(Test_Generator,verbose=0, steps=8)
    test_data, test_label = Test_Generator.next()

    # Correct 함수를 통해 테스트 결과를 우리가 볼 수 있게 출력
    def Correct(data, label, pred) :

        # sort 통해서 predict된 인코딩 중 정답 index 구하기 
        os.chdir(test_path)
        #rd = pred.round(2)
        #print(rd)
        rd = pred.tolist()
        idx1 = rd.index(max(rd))
        sort = sorted(rd, reverse=True)
        sort5 = sort[:5]
        idx2 = rd.index(sort5[1]) 
        idx3 = rd.index(sort5[2]) 
        print('결과')

        # 이미지 출력 
        
        #image = glob.glob(test_path+'/*/*.png')
        #img = Image.open(image[0])
        #plt.imshow(img) 
        # Test Generator로 불러온 이미지 사용 시
        ax = plt.subplot()
        ax.imshow(data) 


        # 인덱스를 통해 해당 유니코드 구한 후 유니코드를 한자로 변환
        unicode_list = os.listdir()
        print('예측한자')
        print(chr(int(unicode_list[idx1])))
        if idx2 == 0.0 :
            pass
        else :
            print(chr(int(unicode_list[idx2])))
        if idx3 == 0.0 :
            pass
        else :
            print(chr(int(unicode_list[idx3])))

    # Correct 함수 실행 및 해당 변수 넣기        
    def num_correct():
        Correct(test_data[0], test_label[0], Predicts[0])
    
    # test 폴더 내부 파일 삭제를 통해 정답 폴더 초기화
    #img.close()
    one_char = test_path + '/*/*.png'
    [os.remove(f) for f in glob.glob(one_char)]

    return num_correct()

# 예시코드
# check_answer()













