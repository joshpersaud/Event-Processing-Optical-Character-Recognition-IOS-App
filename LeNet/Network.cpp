//
//  Network.cpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include "Network.hpp"

void Network:: forward_pass(cube& input, vec& output)
{
    vec flat_input = vectorise(input);
    output = (weights * flat_input) + biases;
    
    this->input = input;
    this->output = output;
}
void Network:: backward_pass(vec& gradient)
{
    vec grad_i_vec = zeros(i_H* i_W * i_D);
    for ( int i = 0; i < (i_H* i_W * i_D); i++)
        grad_i_vec[i] = dot(weights.col(i),gradient);
    
    cube tmp((i_H* i_W * i_D),1,1);
    tmp.slice(0).col(0) = grad_i_vec;
    
    grad_i = reshape(tmp, i_H, i_W, i_D);
    
    accum_grad_w += grad_w;
    
    grad_b = gradient;
    accum_grad_b += grad_b;
    
}
void Network:: update_weights_and_biases(size_t batch_size,double learning_rate)
{
    weights = weights - learning_rate * (accum_grad_w/batch_size);
    biases = biases - learning_rate * (accum_grad_b/batch_size);
    
    reset_accum_grad();
}
cube Network:: get_grad_wrt_i()
{
    return grad_i;
}
mat Network:: get_grad_wrt_w()
{
    return grad_w;
}
vec Network:: get_grad_wrt_b()
{
    return grad_b;
}
mat Network:: get_weights()
{
    return weights;
}
vec Network:: get_biases()
{
    return biases;
}
