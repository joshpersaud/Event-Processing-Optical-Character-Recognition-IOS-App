//
//  LeNet.hpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/14/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//


#include "Convolution.hpp"
#include "Max_Pool.hpp"
#include "Network.hpp"
#include "ReLU.hpp"
#include "Soft_Max.hpp"
#include "Cross_Entropy_Loss_Layer.hpp"

#include <fstream>


class MC_LeNet
{
private:
    vector<cube> training_set;
    vector<vec> training_labels;
    
    vector<cube> convolution1_filters;
    vector<cube> convolution2_filters;
    mat network_weights;
    vec network_biases;
    
    
    size_t t_set_len;
    size_t t_label_len;
    
    
    size_t input_height;
    size_t input_width;
    size_t input_depth;
    size_t filter_height;
    size_t filter_width;
    size_t num_filters;
    size_t filter_vertical_stride;
    size_t filter_horizontal_stride;
    size_t pooling_height;
    size_t pooling_width;
    size_t pooling_vertical_stride;
    size_t pooling_horizontal_stride;
    size_t num_classes;
    size_t batch_size;
    size_t epochs;
    size_t num_batches;
    double learning_rate;
    
    double cum_loss;
    double loss;
    
    Convolution *c1;
    cube c1_out;
    
    ReLU *r1;
    cube r1_out;
    
    Max_Pool *p1;
    cube p1_out;
    
    Convolution *c2;
    cube c2_out;
    
    ReLU *r2;
    cube r2_out;
    
    Max_Pool *p2;
    cube p2_out;
    
    Network *n1;
    vec n1_out;
    
    Soft_Max *s1;
    vec s1_out;
    
    Cross_Entropy_Loss_Layer *e1;

    
    void write_training_results_to_file(string file_path);
    
public:
    
    MC_LeNet(vector<cube> training_set,vector<vec> training_labels,size_t input_height
                                                                    ,size_t input_width
                                                                    ,size_t input_depth
                                                                    ,size_t filter_height
                                                                    ,size_t filter_width
                                                                    ,size_t num_filters
                                                                    ,size_t filter_vertical_stride
                                                                    ,size_t filter_horizontal_stride
                                                                    ,size_t pooling_height
                                                                    ,size_t pooling_width
                                                                    ,size_t pooling_vertical_stride
                                                                    ,size_t pooling_horizontal_stride
                                                                    ,size_t num_classes
                                                                    ,size_t batch_size
                                                                    ,size_t epochs
                                                                    ,double learning_rate)
    {
        this->training_set = training_set;
        this->training_labels = training_labels;
        
        this->input_height = input_height;
        this->input_width = input_width;
        this->input_depth = input_depth;
        this->filter_height = filter_height;
        this->filter_width = filter_width;
        this->num_filters = num_filters;
        this->filter_vertical_stride = filter_vertical_stride;
        this->filter_horizontal_stride = filter_horizontal_stride;
        this->pooling_height = pooling_height;
        this->pooling_width = pooling_width;
        this->pooling_vertical_stride = pooling_vertical_stride;
        this->pooling_horizontal_stride = pooling_horizontal_stride;
        this->num_classes = num_classes;
        this->batch_size = batch_size;
        this->epochs = epochs;
        this->learning_rate = learning_rate;
        
        this->loss = 0.0;
        this->cum_loss = 0.0;
        
        t_set_len = training_set.size();
        t_label_len = training_labels.size();
        
        num_batches = t_set_len/batch_size;
        
        c1 = new Convolution(input_height
                             ,input_width
                             ,input_depth
                             ,filter_height
                             ,filter_width
                             ,filter_vertical_stride
                             ,filter_horizontal_stride
                             ,num_filters);
        
        c1_out = zeros((input_height - filter_height)/filter_vertical_stride + 1
                       ,(input_width - filter_width)/filter_horizontal_stride + 1
                       ,num_filters);
        
        r1 = new ReLU(c1_out.n_rows
                      ,c1_out.n_cols
                      ,num_filters);
        
        r1_out = zeros(c1_out.n_rows
                       ,c1_out.n_cols
                       ,num_filters);
        
        p1 = new Max_Pool(c1_out.n_rows
                          ,c1_out.n_cols
                          ,num_filters
                          ,pooling_height
                          ,pooling_width
                          ,pooling_vertical_stride
                          ,pooling_horizontal_stride);
        
        p1_out = zeros((c1_out.n_rows - pooling_height)/pooling_vertical_stride + 1
                       ,(c1_out.n_cols - pooling_width)/pooling_horizontal_stride + 1
                       ,num_filters);
        
        c2 = new Convolution(p1_out.n_rows
                             ,p1_out.n_cols
                             ,num_filters
                             ,filter_height
                             ,filter_width
                             ,filter_vertical_stride
                             ,filter_horizontal_stride
                             ,num_filters + 10);
        
        c2_out = zeros((p1_out.n_rows - filter_height)/filter_vertical_stride + 1
                       ,(p1_out.n_cols - filter_width)/filter_horizontal_stride + 1
                       ,num_filters + 10);
        r2 = new ReLU(c2_out.n_rows
                      ,c2_out.n_cols
                      ,c2_out.n_slices);
        
        r2_out = zeros(c2_out.n_rows
                       ,c2_out.n_cols
                       ,c2_out.n_slices);
        
        p2 = new Max_Pool(r2_out.n_rows
                        ,r2_out.n_cols
                        ,r2_out.n_slices
                        ,pooling_height
                        ,pooling_width
                        ,pooling_vertical_stride
                        ,pooling_horizontal_stride);
        
        p2_out = zeros((r2_out.n_rows - pooling_height)/pooling_vertical_stride + 1
                       ,(r2_out.n_cols - pooling_width)/pooling_horizontal_stride + 1
                       ,c2_out.n_slices);
        
        n1 = new Network(p2_out.n_rows
                         ,p2_out.n_cols
                         ,p2_out.n_slices
                         ,num_classes);
        
        n1_out = zeros(num_classes);
        
        s1 = new Soft_Max(num_classes);
        
        s1_out = zeros(num_classes);
        
        e1 = new Cross_Entropy_Loss_Layer(num_classes);
        
        
    }
    
    ~MC_LeNet()
    {
        delete c1;
        delete c2;
        delete r1;
        delete r2;
        delete p1;
        delete p2;
        delete n1;
        delete s1;
        delete e1;
    }
    
    void train();
    vec predict(cube img);
    
};


