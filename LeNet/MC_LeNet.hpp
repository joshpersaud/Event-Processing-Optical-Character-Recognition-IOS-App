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

#define I_H_c1 28
#define I_W_c1 28
#define I_D_c1 1

#define I_H_c2 12
#define I_W_c2 12
#define I_D_c2 6

#define F_H 5
#define F_W 5
#define STRIDE_H 1
#define STRIDE_V 1
#define NUM_FILT_f1 6
#define NUM_FILT_f2 16

#define P_H 2
#define P_W 2
#define P_H_STRIDE 2
#define P_V_STRIDE 2

#define LEARNING_RATE .05
#define EPOCHS 10
#define BATCH_SIZE 10


class MC_LeNet
{
private:
    vector<cube> training_set;
    vector<vec> training_labels;
    
    size_t t_set_len;
    size_t t_label_len;
    size_t num_batches;
    
    double cum_loss;
    double accuracy;
    
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

    
    double cross_entropy_loss;
    vec grad_wrt_real_output;
    
    double calculate_cross_entropy_loss(vec real_Label, vec training_Label);
    double calculate_grad_wrt_output(vec real_Label, vec training_Label);
    
public:
    
    MC_LeNet(vector<cube> training_set,vector<vec> training_labels)
    {
        this->training_set = training_set;
        t_set_len = training_set.size();
        
        this->training_labels = training_labels;
        t_label_len = training_labels.size();
        
        this->num_batches = t_set_len/BATCH_SIZE;
        
        cross_entropy_loss = 0.0;
        grad_wrt_real_output = 0.0;
        
        c1 = new Convolution(I_H_c1
                             ,I_W_c1
                             ,I_D_c1
                             ,F_H
                             ,F_W
                             ,STRIDE_V
                             ,STRIDE_H
                             ,NUM_FILT_f1);
        
        c1_out = zeros(I_H_c1 - (F_H - 1)
                       ,I_H_c1 - (F_W - 1)
                       ,NUM_FILT_f1);
        
        r1 = new ReLU(I_H_c1 - (F_H - 1)
                      ,I_H_c1 - (F_W - 1)
                      ,NUM_FILT_f1);
        
        r1_out = zeros(I_H_c1 - (F_H - 1)
                       ,I_H_c1 - (F_W - 1)
                       ,NUM_FILT_f1);
        
        p1 = new Max_Pool(I_H_c1 - (F_H - 1)
                          ,I_H_c1 - (F_W - 1)
                          ,NUM_FILT_f1
                          ,P_H
                          ,P_W
                          ,P_V_STRIDE
                          ,P_H_STRIDE);
        
        p1_out = zeros(I_H_c2
                       ,I_W_c2
                       ,I_D_c2);
        
        c2 = new Convolution(I_H_c2
                             ,I_W_c2
                             ,I_D_c2
                             ,F_H
                             ,F_W
                             ,STRIDE_V
                             ,STRIDE_H
                             ,NUM_FILT_f2);
        
        c2_out = zeros(I_H_c2 - (F_H - 1)
                       ,I_H_c2 - (F_W - 1)
                       ,NUM_FILT_f2);
        
        r2 = new ReLU(I_H_c2 - (F_H - 1)
                      ,I_H_c2 - (F_W - 1)
                      ,NUM_FILT_f2);
        
        r2_out = zeros(I_H_c2 - (F_H - 1)
                      ,I_H_c2 - (F_W - 1)
                      ,NUM_FILT_f2);
        
        p2 = new Max_Pool(I_H_c2 - (F_H - 1)
                          ,I_H_c2 - (F_W - 1)
                          ,NUM_FILT_f2
                          ,P_H
                          ,P_W
                          ,P_V_STRIDE
                          ,P_H_STRIDE);
        
        p2_out = zeros((I_H_c2 - (F_H - 1)) / 2
                       ,(I_H_c2 - (F_W - 1)) / 2
                       ,NUM_FILT_f2);
        
        n1 = new Network((I_H_c2 - (F_H - 1)) / 2
                         ,(I_H_c2 - (F_W - 1)) / 2
                         ,NUM_FILT_f2
                         ,t_label_len);
        
        n1_out = zeros(t_label_len);
        
        s1 = new Soft_Max(t_label_len);
        
        s1_out = zeros(t_label_len);
        
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
    }
    
    void train();
    void predict(cube img);
    
    
};
