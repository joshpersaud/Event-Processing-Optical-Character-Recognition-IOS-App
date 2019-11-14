//
//  Convolution.hpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include <iostream>
#include <armadillo>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>

using namespace arma;
using namespace std;

class Convolution
{
private:
    cube input;
    size_t i_H;
    size_t i_W;
    size_t i_D;
    
    vector<cube> filters;
    size_t f_H;
    size_t f_W;
    size_t f_v_stride;
    size_t f_h_stride;
    size_t num_filters;
    
    cube output;
    
    cube grad_i;
    cube accum_grad_i;
    
    vector<cube> grad_f;
    vector<cube> accum_grad_f;
    
    
    double assign_rand_filt(double mean, double variance)
    {
        /*
         
         Using a normal distribution given a mean and standard deviation generate a random number
         which falls within the 2 standard deviations.
         
         */
        double std_dev = sqrt(variance);
        
        random_device rand_dev;
        mt19937 gen(rand_dev());
        normal_distribution<> dist(mean,std_dev);
        
        double rand = (3*std_dev);
        
        while((rand - mean) > (2.0 * std_dev))
        {
            rand = dist(gen);
        }
        
        return rand;
    }
    
    
    
    void reset_accum_grad()
    {
        /*
         
         Reset accumulated gradient filters back to 0
         This way the next epochs backpropagation is not skewed
         
         */
        
        accum_grad_i = zeros(i_H,i_W,i_D);
        
        accum_grad_f.clear();
        accum_grad_f.resize(num_filters);
        
        for ( int i = 0; i < num_filters; i++)
        {
            accum_grad_f[i] = zeros(f_H,f_W,i_D);
        }
    }
    
public:
    
    Convolution(size_t i_H
                ,size_t i_W
                ,size_t i_D
                ,size_t f_H
                ,size_t f_W
                ,size_t f_v_stride
                ,size_t f_h_stride
                ,size_t num_filters)
    {
        this->i_H = i_H;
        this->i_W = i_W;
        this->i_D = i_D;
        
        this->f_H = f_H;
        this->f_W = f_W;
        this->f_v_stride = f_v_stride;
        this->f_h_stride = f_h_stride;
        
        this->num_filters = num_filters;
        
        assert((i_H - f_H) % f_v_stride == 0);
        assert((i_W - f_W) % f_h_stride == 0);
        
        this->filters.resize(num_filters);
        for ( auto& filt : filters)
        {
            filt = zeros(f_H,f_W,i_D);
            filt.imbue( [&](){return assign_rand_filt(0.0, 1.0);} );
        }
        
    }
    
    void foward_pass(cube& input, cube& output);
    void backward_pass(cube& gradient);
    void update_filters(size_t batch_size, double learning_rate);
    vector<cube> get_filters();
    cube get_grad_wrt_i();
    vector<cube> get_grad_wrt_f();
    
};

