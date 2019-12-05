#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:08:11 2019

@author: danielvilajeti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:44:17 2019

@author: danielvilajeti
"""

import cv2 as cv
import os
import numpy as np


def create_categorical_map(unique_labels):
    
    labels_dict = {}
    new_list = unique_labels
    new_list.sort()
    
    index = 0
    
    for elem in new_list:
        labels_dict[elem] = index
        index += 1
        
    return labels_dict
    
def to_categorical(label,cat_map):
    num_classes = len(cat_map)
    
    new_label = np.zeros((num_classes))
    new_label[cat_map[label]] = 1
    
    return new_label
    
def normalize_img(img):
    
    return (abs(img - 255)).astype('float32')/255.0


def load_train_data():
    
    Training_DIR = './Dataset/Training_Data'
    
    str_to_char  = {'dot' : '.' 
                    , 'colon' : ':'
                    , 'or': '|'
                }

    images_loaded = 0

    unique_labels = []

    train_imgs = []
    train_labels = []
    
    for img_folder in os.listdir(Training_DIR):
        
        if img_folder != '.DS_Store':
            
            try:
                label = ord(str_to_char[img_folder])
                
            except KeyError:
               
                if len(img_folder) == 2:
                    label = ord(img_folder[1])
                else:            
                    label = ord(img_folder)
        
            unique_labels.append(label)
        
            img_path = Training_DIR + '/' + img_folder + '/'
        
            for img_file in os.listdir(img_path):
                images_loaded += 1
                
                if img_file != '.DS_Store':
                    img = cv.imread(img_path + img_file,0)
                    
                    train_imgs.append(img)
                    train_labels.append(label)
    
                     
    return (train_imgs,train_labels,unique_labels)

                
def load_test_data():
    
    Testing_DIR = './Dataset/Testing_Data'
    
    str_to_char  = {'dot' : '.' 
                    , 'colon' : ':'
                    , 'or': '|'
                }

    images_loaded = 0

    test_imgs = []
    test_labels = []
    
    for img_folder in os.listdir(Testing_DIR):
        if img_folder != '.DS_Store':
            
            try:
                label = ord(str_to_char[img_folder])
                
            except KeyError:
               
                if len(img_folder) == 2:
                    label = ord(img_folder[1])
                else:            
                    label = ord(img_folder)
        
            img_path = Testing_DIR + '/' + img_folder + '/'

            for img_file in os.listdir(img_path):
                
                images_loaded += 1
                
                if img_file != '.DS_Store':
                    img = cv.imread(img_path + img_file,0)
                    
                    test_imgs.append(img)
                    test_labels.append(label)
                
    return (test_imgs,test_labels)
    


def get_train_data(train_imgs,train_labels,cat_map):
    
    full_train_imgs = []
    full_train_labels = []
    
    for i in range(len(train_imgs)):
        full_train_imgs.append(normalize_img(train_imgs[i]))
        full_train_labels.append(to_categorical(train_labels[i],cat_map))
        
    return (full_train_imgs,full_train_labels)
        

def get_test_data(test_imgs,test_labels,cat_map):

    full_test_imgs = []
    full_test_labels = []

    for i in range(len(test_imgs)):
        full_test_imgs.append(normalize_img(test_imgs[i]))
        full_test_labels.append(to_categorical(test_labels[i],cat_map))
        
    return (full_test_imgs,full_test_labels)
 
def get_data():
    
    load_train_data()
    
    (train_imgs,train_labels,unique_labels) = load_train_data()
    (test_imgs,test_labels) = load_test_data()

    cat_map = create_categorical_map(unique_labels)
    
    (train_imgs_data,train_labels_data) = get_train_data(train_imgs,train_labels,cat_map)
    (test_imgs_data,test_labels_data) = get_test_data(test_imgs,test_labels,cat_map)
    
    tr_set_len = len(train_imgs_data)
    te_set_len = len(test_imgs_data)
    
    train_imgs_data = np.asarray(train_imgs_data)
    train_imgs_data = train_imgs_data.reshape(tr_set_len,28,28,1)
    train_labels_data = np.asarray(train_labels_data)
    
    test_imgs_data = np.asarray(test_imgs_data)
    test_imgs_data = test_imgs_data.reshape(te_set_len,28,28,1)
    test_labels_data = np.asarray(test_labels_data)
    
    training_data = [train_imgs_data,train_labels_data]
    testing_data = [test_imgs_data,test_labels_data]
    
    unique_labels.sort()
    
    return (training_data,testing_data,unique_labels)




      