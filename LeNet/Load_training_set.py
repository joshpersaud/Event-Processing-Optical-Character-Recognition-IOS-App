#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:44:17 2019

@author: danielvilajeti
"""

import cv2 as cv
import os
import csv

def normalize(img):
    for i in range(28):
    
        for j in range(28):
        
            if (img[i][j] < (int)(255/2)):
                img[i][j] = 255 - img[i][j]
            
            else:
                img[i][j] = 0 + (255 - img[i][j])
        
    return img        

TRAIN_IMGS_FILE_NAME = 'training_set.csv'
TRAIN_LABELS_FILE_NAME = 'training_labels.csv'
DATA_DIR = './Dataset'


str_to_char  = {'dot' : '.' 
                , 'colon' : ':'
                , 'or': '|'
            }

images_loaded = 0
labels = []
with open(TRAIN_IMGS_FILE_NAME,mode = 'w') as img_file:
            
    img_file_writer = csv.writer(img_file,delimiter=' ',quoting=csv.QUOTE_NONE)
    
    for img_folder in os.listdir(DATA_DIR+'/Training Data/'):
        if img_folder != '.DS_Store':
            
            try:
                label = (int)(img_folder)
                
            except ValueError:
                
                try:
                    label = ord(str_to_char[img_folder])
        
                except KeyError:
                    
                    if len(img_folder) == 2:
                        label = ord(img_folder[1])
                    else:            
                        label = ord(img_folder)
        
            labels.append(label)
            
            img_path = DATA_DIR+'/Training Data/' + img_folder + '/'

            for img_file in os.listdir(img_path):

                images_loaded += 1
                
                if img_file != '.DS_Store':
                    img = cv.imread(img_path + img_file,0)
                    
                    img = normalize(img)
                    
                    img_as_arr = []
                    img_as_arr.append(label)

                    for row in img:
                        for val in row:
                            img_as_arr.append(val)
        
                    img_file_writer.writerow(img_as_arr)
                    print('SUCCESS!')
      
labels.sort()
print(labels)
print(len(labels))
print('TRAINING SET UPLOADED!!!')
print('Number of images loaded: ' + str(images_loaded))