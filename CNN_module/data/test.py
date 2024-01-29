import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from matplotlib.font_manager import findfont, FontProperties

from settings import ImageSize
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import sys
import json

def test_model(model_path, test_path):
    CNN = tf.keras.models.load_model(model_path)
    Test_Data_Generator = ImageDataGenerator(rescale=1./255)
    Test_Generator = Test_Data_Generator.flow_from_directory(test_path,
                                                            target_size=ImageSize,
                                                            shuffle=False,
                                                            class_mode='categorical')
    # Evaluate the model on the test data
    test_loss, test_accuracy = CNN.evaluate(Test_Generator)

    # Predict the labels for the test data
    predictions = CNN.predict(Test_Generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get the true labels of the test data
    true_classes = Test_Generator.classes

    # Calculate additional evaluation metrics
    precision = precision_score(true_classes, predicted_classes, average='macro')
    recall = recall_score(true_classes, predicted_classes, average='macro')
    f1 = f1_score(true_classes, predicted_classes, average='macro')
    confusion = confusion_matrix(true_classes, predicted_classes)
    
    f = open('/mnt/d/download/test_result/result.txt', 'w')
    
    print("Test Loss:", test_loss, file = f)
    print("Test Accuracy:", test_accuracy, file = f)
    print("Precision:", precision, file = f)
    print("Recall:", recall, file = f)
    print("F1 Score:", f1, file = f)
    # print("Confusion Matrix:")
    # print(confusion)
    
    f.close()
    
    # Plotting the evaluation metrics
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [test_accuracy, precision, recall, f1]

    plt.bar(labels, values)
    plt.title('CNN Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.savefig('/mnt/d/download/test_result/savefig_result.png')

    # Calculate accuracy for each character
    character_accuracy = {}
    total_occurrences = np.bincount(true_classes)
    for true_label, predicted_label in zip(true_classes, predicted_classes):
        if true_label == predicted_label:
            character_accuracy[true_label] = character_accuracy.get(true_label, 0) + 1

    # Normalize accuracy by dividing by the total number of occurrences of each character
    for label, count in Test_Generator.class_indices.items():
        character_accuracy[count] = character_accuracy.get(count, 0) / total_occurrences[count]

    # Create a mapping from class labels to folder names
    label_to_folder = {v: k for k, v in Test_Generator.class_indices.items()}

    # Calculate accuracy for each folder
    folder_accuracy = {}
    for label, accuracy in character_accuracy.items():
        if label in label_to_folder:
            folder_name = label_to_folder[label]
            folder_accuracy[folder_name] = accuracy

    # Sort folder accuracy in descending order
    sorted_folder_accuracy = dict(sorted(folder_accuracy.items(), key=lambda x: x[1], reverse=True))
    
    json_file = 'sorted_folder_accuracy.json'
    
    with open('/mnt/d/download/test_result/' + json_file, 'w') as f:
        json.dump(sorted_folder_accuracy, f)