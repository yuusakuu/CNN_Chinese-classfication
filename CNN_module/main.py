from data import load_gen
from data import run
from data import test
from model import CNN_model


if __name__=="__main__":

    data_path = "/mnt/d/download/fdata"
    print('data_load')
    Train, Val = load_gen.gener(data_path)

    print('show model')
    print(CNN_model.loadCNN())

    print('run_model')
    file_name = "CNN_test.h5"
    save_path = "/mnt/d/download/model"
    Train_Accuracy, Val_Accuracy, Train_Loss, Val_Loss, epochs_range = run.run_model(Train, Val, file_name, save_path)

    print('test model')
    model_path = '/mnt/d/download/최종 모델/Final_model_v7-20230622T045322Z-001/Final_model_v7'
    test_path = '/mnt/d/testset'
    test.test_model(model_path, test_path)