import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from matplotlib import pyplot as plt

def check_answer(model_path, test_path, idx):
    
    # model_path = 'D:/fast_test_model'
    os.chdir(model_path)

    # Load the trained model
    CNN = tf.keras.models.load_model('CNN_Model.h5')

    #test_path = 'D:/testset_fast_test'

    Image_Size = (50, 50)

    Test_Data_Genetor = ImageDataGenerator( rescale=1./255 )
    Test_Generator = Test_Data_Genetor.flow_from_directory( test_path,
                                                            target_size = Image_Size,
                                                            shuffle = False,
                                                            class_mode = 'categorical' )
    
    Test_Generator.reset()

    Predicts=CNN.predict(Test_Generator,verbose=0, steps=8)
    
    test_data, test_label = Test_Generator.next()

    cnt_list = list(range(1, len(test_data)+1))
    
    def Correct(cnt, data, label, pred) :
        # cnt = 문제 번호

        rd = pred.round(1)
        #print(rd)
        rd = rd.tolist()
        idx = rd.index(max(rd))
        label = label.tolist()
        label_idx = label.index(max(label))
        print(f'{cnt}번 문제')
        ax = plt.subplot( )   
        ax.imshow(data)
        plt.show() 
        if label_idx == idx:
            print('정답')
            cnt += 1
            #print('=================================================================')
        else :
            print('오답')
            cnt += 1
            #print('=================================================================')


    def num_correct(idx):
        Correct(cnt_list[idx], test_data[idx], test_label[idx], Predicts[idx])

    return num_correct(idx)



# 예시 입력문
#model_path = 'D:/fast_test_model'
#test_path = 'D:/testset_fast_test'
#idx = 2


#check_answer(model_path, test_path, idx)