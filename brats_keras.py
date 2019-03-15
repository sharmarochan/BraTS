# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:06:37 2019

@author: Rochan.Sharma

major source

https://github.com/zsdonghao/u-net-brain-tumor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import os
from glob import glob
import sys
import random
from tqdm import tqdm_notebook
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.metrics import jaccard_similarity_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable
from tqdm import tqdm
import os
import nibabel as nib
import pickle


data_set = r"C:\BraTS\Brats17TrainingData\HGG"


#get the name of the folder_names
folder_names_arr = []
folder_path_arr = []

for (dirpath, dirnames, filenames) in os.walk(data_set):
    for f in filenames:
        full_path = (os.path.join(dirpath, f))
        patient_file_dir = os.path.dirname(full_path)
        folder_path_arr.append(patient_file_dir)
        folder_names_arr.append(os.path.basename(patient_file_dir))
        



data_types = ['flair', 't1', 't1ce', 't2']


file_type_flair = []
file_type_seg = []
file_type_t1 = []
file_type_t2 = []
file_type_t1ce = []

count = 0

print(len(folder_path_arr))



for i in folder_path_arr:
    print("folder path", i)
    files = os.listdir(i)           #getting all the files from a single patient folder
    print(files)
    
    for f in files:
        f_split_with_ext = f.split("_")[-1]
        file_type = f_split_with_ext.split(".")[0]
        print("-----------insertion start------------------")
        
#        print(file_type)                            #flair, seg, t1, t1ce, t2
        
        
                    
        if file_type == "flair":
            flair_file = os.path.join(i,f)
            print(flair_file)
            img_flair = nib.load(flair_file).get_data()
#            print(img.shape)
            file_type_flair.append(img_flair)
            
        if file_type == "seg":
            seg_file = os.path.join(i,f)
            print(seg_file)
            img_seg = nib.load(seg_file).get_data()
#            print(img.shape)
            file_type_seg.append(img_seg)
            
        if file_type == "t1":
            t1_file = os.path.join(i,f)
            print(t1_file)
            img_t1 = nib.load(t1_file).get_data()
#            print(img.shape)
            file_type_t1.append(img_t1)
            
        if file_type == "t1ce":
            t1ce_file = os.path.join(i,f)
            print(t1ce_file)
            img_t1ce = nib.load(t1ce_file).get_data()
#            print(img.shape)
            file_type_t1ce.append(img_t1ce)
            
        if file_type == "t2":
            t2_file = os.path.join(i,f)
#            print(t2_file)
            img_t2 = nib.load(t2_file).get_data()
#            print(img.shape)
            file_type_t2.append(img_t2)
            

#        print(len(file_type_flair))
#        print(len(file_type_seg))
#        print(len(file_type_t1))
#        print(len(file_type_t2))
#        print(len(file_type_t1ce))
        
        print("-----------insertion stop------------------")


    count = count + 1
    print("count:", count)
#    
#    if count == 10:
#        break
        
        



m_flair = np.mean(file_type_flair)                         
s_flair = np.std(file_type_flair)

m_t1 = np.mean(file_type_t1)                         
s_t1 = np.std(file_type_t1)

m_t1ce = np.mean(file_type_t2)                         
s_t1ce = np.std(file_type_t2)

m_t2 = np.mean(file_type_t1ce)                         
s_t2 = np.std(file_type_t1ce)



del file_type_flair
del file_type_seg
del file_type_t1
del file_type_t2
del file_type_t1ce




print(m_flair, s_flair, m_t1, s_t1, m_t1ce, s_t1ce, m_t2, s_t2)

with open(data_set + 'mean_std_dict.pickle', 'wb') as f:
    pickle.dump((m_flair, s_flair, m_t1, s_t1, m_t1ce, s_t1ce, m_t2, s_t2), f, protocol=4)



m_flair = 70.32705375911777
s_flair = 564.209178073765
m_t1 = 95.34458429423694
s_t1 = 521.8619904432137
m_t1ce = 105.57924098558448
s_t1ce = 609.0591653796687
m_t2 = 107.23378603172357
s_t2 = 541.411026439475

#####################################################################
################data pre-processing#################################
#####################################################################   
#    


count = 0

#X_train_input = np.array([])
#X_train_target = np.array([])

X_train_input = []
X_train_target = []



for arr in folder_path_arr:
    count = count + 1
    print("patient count", count)
    
    if count == 40:
        break
    
    
    all_3d_data = []
    print("folder path", arr)
    files = os.listdir(arr)      #getting all the files from a single patient folder
#    print(files)
    
    
    for f in files:
        f_split_with_ext = f.split("_")[-1]
        file_type = f_split_with_ext.split(".")[0]
        
#        print(file_type)                            #flair, seg, t1, t1ce, t2
        
        if file_type == "flair":
            flair_file = os.path.join(arr,f)
#            print(flair_file)
            img_flair = nib.load(flair_file).get_data()
            img_flair = (img_flair - m_flair) / s_flair
            img_flair = img_flair.astype(np.float32)
            all_3d_data.append(img_flair)
            
            
        if file_type == "t1":
            t1_file = os.path.join(arr,f)
#            print(t1_file)
            img_t1 = nib.load(t1_file).get_data()
            img_t1 = (img_t1 - m_t1) / s_t1
            img_t1 = img_t1.astype(np.float32)
            all_3d_data.append(img_t1)
            
        if file_type == "t1ce":
            t1ce_file = os.path.join(arr,f)
#            print(t1ce_file)
            img_t1ce = nib.load(t1ce_file).get_data()
            img_t1ce = (img_t1ce - m_t1ce) / s_t1ce
            img_t1ce = img_t1ce.astype(np.float32)
            all_3d_data.append(img_t1ce)
            
        if file_type == "t2":
            t2_file = os.path.join(arr,f)
#            print(t2_file)
            img_t2 = nib.load(t2_file).get_data()
            img_t2 = (img_t2 - m_t2) / s_t2
            img_t2 = img_t2.astype(np.float32)
            all_3d_data.append(img_t2)
            
            
        if file_type == "seg":
            seg_file = os.path.join(arr,f)
#            print(seg_file)
            seg_file = nib.load(seg_file).get_data()
            seg_file = np.transpose(seg_file, (1, 0, 2))
            
#    

    

#print(len(all_3d_data))
    count_split = 0
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
#        print(combined_array.shape)
        combined_array = np.transpose(combined_array, (1, 0, 2))
#        print(combined_array.shape)
        combined_array.astype(np.float32)
#        X_train_input = np.append(X_train_input, combined_array, axis=0)
        X_train_input.append(combined_array)
        count_split = count_split +1

        
        
        seg_2d = seg_file[:, :, j]
        seg_2d.astype(int)
        seg_2d_expnd = np.expand_dims(seg_2d, axis=-1)
#        print(seg_2d_expnd.shape)
        X_train_target.append(seg_2d_expnd)
#        X_train_target = np.append(X_train_target, seg_2d, axis=0)
        
    print("count_split ", count_split)
        



#del all_3d_data
#



print(len(X_train_input))
print(len(X_train_target))


print(type(X_train_input))
print(type(X_train_target))


X_train_input_arr = np.asarray(X_train_input)
X_train_target_arr = np.asarray(X_train_target)



del X_train_input
del X_train_target



print(type(X_train_input_arr))
print(type(X_train_target_arr))



print(len(X_train_input_arr))
print(len(X_train_target_arr))


print("[INFO] data matrix: {:.2f}MB".format(X_train_input_arr.nbytes / (1024 * 1000.0)))



print("[INFO] data matrix: {:.2f}MB".format(X_train_target_arr.nbytes / (1024 * 1000.0)))




###################################################################################################################
###################################################################################################################
###################################################################################################################
# https://www.depends-on-the-definition.com/unet-keras-segmenting-images/

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img






print(X_train_input_arr.shape)
print(X_train_target_arr.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_input_arr, X_train_target_arr, test_size=0.2, random_state=2018)

print(len(X_train))
print(len(X_valid))
print(len(y_train))
print(len(y_valid))





del X_train_input_arr
del X_train_target_arr





def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x






def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    print(input_img.shape)
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


im_height = 240
im_width = 240

input_img = Input((im_height, im_width, 4), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()









callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('brats_challenge.h5', verbose=1, save_best_only=True, save_weights_only=True)
]






results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))







model.load_weights('brats_challenge.h5')







model.evaluate(X_valid, y_valid, verbose=1)





preds_train = model.predict(X_train, verbose=1)

preds_val = model.predict(X_valid, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)





