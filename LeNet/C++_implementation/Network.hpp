//
//  Network.hpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include <iostream>
#include <armadillo>
#include <cassert>
#include <cmath>
#include <random>

using namespace arma;
using namespace std;

class Network
{
private:
    cube input;
    size_t i_H;
    size_t i_W;
    size_t i_D;
    
    
    size_t numOutputs;
    vec output;
    
    mat weights;
    vec biases;
    
    cube grad_i;
    mat grad_w;
    vec grad_b;
    
    cube accum_grad_i;
    mat accum_grad_w;
    vec accum_grad_b;
    
    double assign_rand_weights(double mean, double variance)
    {
        /*
         
         Using a normal distribution given a mean and standard deviation generate a random number
         which falls within the 2 standard deviations.
         
         */
        double std_dev = sqrt(variance);
        mat candidate = {3.0 * std_dev};
    
        while (std::abs(candidate[0] - mean) > 2.0 * std_dev)
          candidate.randn(1, 1);
        return candidate[0];
    }
    
    void reset_accum_grad()
    {
        /*
         
         Reset accumulated gradient filters back to 0
         This way the next epochs backpropagation is not skewed
         
         */
        
        accum_grad_i = zeros(i_H,i_W,i_D);
        accum_grad_w = zeros(numOutputs,(i_H * i_W * i_D));
        accum_grad_b = zeros(numOutputs);
    }
    
public:
    
    Network(size_t i_H
            ,size_t i_W
            ,size_t i_D
            ,size_t numOutputs)
    {
        this->i_H = i_H;
        this->i_W = i_W;
        this->i_D = i_D;
        
        this->numOutputs = numOutputs;
        
        weights = zeros(numOutputs,(i_H * i_W * i_D));
        weights.imbue( [&](){return assign_rand_weights(0.0,1.0);} );
        biases = zeros(numOutputs);
        
        reset_accum_grad();
    }
    
    void forward_pass(cube& input, vec& output);
    void backward_pass(vec& gradient);
    void update_weights_and_biases(size_t batch_size,double learning_rate);
    cube get_grad_wrt_i();
    mat get_grad_wrt_w();
    vec get_grad_wrt_b();
    mat get_weights();
    vec get_biases();
    
};
