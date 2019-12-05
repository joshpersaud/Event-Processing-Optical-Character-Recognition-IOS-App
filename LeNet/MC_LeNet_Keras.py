#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:58:58 2019

@author: danielvilajeti
"""

import keras

from Load_training_set2 import get_data

import coremltools


class MC_LeNet:
    
    def __init__(self,img_h,img_w,img_d,num_classes,activation='relu',weights_path=None):
        
        self.img_h = img_h
        self.img_w = img_w
        self.img_d = img_d
        self.num_classes = num_classes
        self.activation = activation
        
        self.input_shape = (img_h,img_w,img_d)
        self.num_classes = num_classes
        
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.convolutional.Conv2D(filters=6
                              ,kernel_size=5
                              ,input_shape=self.input_shape
                              ,strides=(1,1)
                              ,padding='same'
                              ,data_format= 'channels_last'))
        self.model.add(keras.layers.core.Activation(self.activation))
        self.model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        self.model.add(keras.layers.convolutional.Conv2D(filters=16
                              ,kernel_size=5
                              ,strides=(1,1)
                              ,padding='same'
                              ,data_format= 'channels_last'))
        
        self.model.add(keras.layers.core.Activation(self.activation))
        self.model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        self.model.add(keras.layers.core.Flatten())
        self.model.add(keras.layers.core.Dense(1000))
        self.model.add(keras.layers.core.Activation(activation))
		
        self.model.add(keras.layers.core.Dense(self.num_classes))
        
        self.model.add(keras.layers.core.Activation("softmax"))
        
        if weights_path is not None:
            self.model.load_weights(weights_path)
        
		
        
    def get_model(self):
        return self.model
        
        
def main():
    
    (train_data,test_data,unique_labels) = get_data()
    
    cnn = MC_LeNet(28,28,1,93)
    cnn_model = cnn.get_model()
    
    opt = keras.optimizers.Adam(lr= .01)
    cnn_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    cnn_model.fit(x=train_data[0],y=train_data[1],batch_size=1723, epochs=10, verbose=1)
    (loss, accuracy) = cnn_model.evaluate(x=test_data[0],y=test_data[1],batch_size=989,verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
    
    cnn_model.save('/Users/danielvilajeti/Documents/COMPUTER_SCIENCE/CAPSTONE/LeNet_CoreML/model.h5')
    
    output_labels = []
    
    for val in unique_labels:
        output_labels.append(chr(val))
 
    print(keras.__version__)
   
    keras_m  = coremltools.converters.keras.convert('/Users/danielvilajeti/Documents/COMPUTER_SCIENCE/CAPSTONE/LeNet_CoreML/model.h5')
    
    keras_m.save('./MC_LeNet.mlmodel')
    
     
main()
     
    
