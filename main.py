import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import cv2
from os import listdir

#CNN
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dense

from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2
from keras.regularizers import l1
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras import backend as K

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print()
import time
print("Timer started\n")
print("-----------------------------------------------------------------------------------------")
start = time.time()

import ray
import psutil
ray.init(num_gpus = 1, ignore_reinit_error=True)

EPOCHS = 80 #iterations
INIT_LR = 0.001 #learning rate

BS = 32 #batch size

default_image_size = tuple((100, 100))
directory_root = '../input/flowers-recognition/'

@ray.remote(num_gpus = 1)
def train_fit():
    
    def convert_image_to_array(image_dir):
        try:
            image = cv2.imread(image_dir)
            if image is not None :
                image = cv2.resize(image, default_image_size)   
                return img_to_array(image)
            else :
                return np.array([])
        except Exception as e:
            print(f"Error : {e}")
            return None
    image_list, label_list = [], []
    
    try:
        print("[INFO] Loading images ...")
        root_dir = listdir(directory_root)
        for directory in root_dir :
            # remove .DS_Store from list
            if directory == ".DS_Store" :
                root_dir.remove(directory)

        for flower_folder in root_dir :
            flower_name_folder_list = listdir(f"{directory_root}/{flower_folder}")

            for flower_name_folder in flower_name_folder_list :
                # remove .DS_Store from list
                if flower_name_folder == ".DS_Store" :
                    flower_name_folder_list.remove(flower_name_folder)

            for flower_name_folder in flower_name_folder_list:
                print(f"[INFO] Processing {flower_name_folder} ...")
                flower_name_image_list = listdir(f"{directory_root}/{flower_folder}/{flower_name_folder}/")

                for single_flower_name_image in flower_name_image_list :
                    if single_flower_name_image == ".DS_Store" :
                        flower_name_image_list.remove(single_flower_name_image)
                k=0
                for image in flower_name_image_list:
                    k=k+1
                    image_directory = f"{directory_root}/{flower_folder}/{flower_name_folder}/{image}"
                    if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                        image_list.append(convert_image_to_array(image_directory))
                        label_list.append(flower_name_folder)
                        
        print("[INFO] Image loading completed")  
    except Exception as e:
        print(f"Error : {e}")
        
    image_size = len(image_list)
    label_binarizer = preprocessing.LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)
    pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
    n_classes = len(label_binarizer.classes_)

    np_image_list = np.array(image_list, dtype=np.float16) / 225.0
    
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.20) 
    
    aug = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.1,zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    aug.fit(x_train)
    
    def create_and_fit_model():
        
        inputShape = (100, 100, 3)
        
        if K.image_data_format() == "channels_first":
            inputShape = (3, 100, 100)
            
        model = Sequential()
        
        model.add(Conv2D(filters = 20, kernel_size = (5,5),strides=(1,1),padding="valid",
                         activation = 'relu', input_shape = inputShape, kernel_initializer="glorot_uniform",
               kernel_regularizer=l2(INIT_LR), bias_regularizer=l2(INIT_LR)))
          
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
        model.add(Conv2D(filters = 50, kernel_size = (5,5),padding="valid",activation = 'relu',kernel_initializer="glorot_uniform",
               kernel_regularizer=l2(INIT_LR), bias_regularizer=l2(INIT_LR)))
        
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Flatten())
        
        model.add(Dense(units = 500, activation = 'relu'))
        
        model.add(Dense(5,activation ='softmax'))
        
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        #decay=INIT_LR / EPOCHS
        #beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
       
        model.compile(optimizer = opt,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy']) 
        
        his=model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),
        validation_data=(x_test, y_test),
        steps_per_epoch=x_train.shape[0] // BS,
        epochs=EPOCHS, verbose=2)
        
        Y_pred = model.predict(x_test)
        y_pred=np.argmax(Y_pred,axis=1)
        Y_test = np.argmax(y_test,axis=1)
        
        print("\n")
        print("====================CLASSIFICATION REPORT====================")
        print()
        target_names = ['dandelion','daisy','sunflower','tulip','rose']
        print(classification_report(Y_test, y_pred,target_names=target_names))
        print("\n")
        
        print("====================EVALUATION METRICS=======================")
        print()
        #no of classes
        print("# of classes  :",n_classes)
        # precision tp / (tp + fp)
        precision = precision_score(Y_test, y_pred,average='macro')
        print('Precision     : %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(Y_test, y_pred,average='macro')
        print('Recall        : %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(Y_test, y_pred,average='macro')
        print('F1 score      : %f' % f1)
        scores_train = model.evaluate(x_train, y_train,verbose=0)
        scores_test = model.evaluate(x_test, y_test,verbose=0)
        
        print(f"Train Accuracy: {scores_train[1]*100}")
        print(f"Test Accuracy : {scores_test[1]*100}")
        print("\n")
        
        print("====================CONFUSION MATRIX=========================")
        print()
        
        def cm2df(cm, labels):
            df = pd.DataFrame()
    # rows
            for i, row_label in enumerate(labels):
                rowdata={}
        # columns
                for j, col_label in enumerate(labels): 
                    rowdata[col_label]=cm[i,j]
                df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
            return df[labels]

        cm = confusion_matrix(Y_test,y_pred).reshape((5, 5))
        df = cm2df(cm, ['dandelion','daisy','sunflower','tulip','rose'])
        print(df)
        
    create_and_fit_model()
    
output = train_fit.remote()

ray.get(output)
print()
#print("--------------------------------------------------------------------------------------------")
end = time.time()
