//
//  ReLU.cpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include "ReLU.hpp"
void ReLU:: forward_pass(cube& input, cube& output)
{
    output = zeros(i_H,i_W,i_D);
    output = arma::max(input,output);
    
    this->input = input;
    this->output = output;
}
void ReLU:: backward_pass(cube gradient)
{
    grad_wrt_i = input;
    grad_wrt_i.transform([](double val){return val > 0? 1 : 0;});
    
    grad_wrt_i = grad_wrt_i % gradient;
}
cube ReLU:: get_grad_wrt_i()
{
    return grad_wrt_i;
}
