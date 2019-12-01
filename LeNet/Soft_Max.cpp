//
//  Soft_Max.cpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include "Soft_Max.hpp"

void Soft_Max:: forward_pass(vec& input,vec& output)
{
    
    double sumExp = arma::accu(arma::exp(input - arma::max(input)));
    output = arma::exp(input - arma::max(input))/sumExp;
    
    this->input = input;
    this->output = output;
    
}
void Soft_Max:: backward_pass(vec& gradient)
{
    double sub = dot(gradient,output);
    grad_wrt_i = (gradient - sub) % output;
}
vec Soft_Max:: get_grad_wrt_i()
{
    return grad_wrt_i;
}

