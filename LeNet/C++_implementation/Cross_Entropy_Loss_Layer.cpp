//
//  Cross_Entropy_Loss_Layer.cpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/30/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include "Cross_Entropy_Loss_Layer.hpp"

double Cross_Entropy_Loss_Layer:: forward_pass(vec& real_label, vec& predicted_label)
{
    assert(predicted_label.n_elem == num_inputs);
    assert(real_label.n_elem == num_inputs);
    
    this->predicted_label = predicted_label;
    this->real_label = real_label;
    
    this->loss = -dot(real_label,arma::log(predicted_label));
    
    return this->loss;
}
void Cross_Entropy_Loss_Layer:: backward_pass()
{
    grad_wrt_predicted_label = -(real_label % (1/predicted_label));
}
vec Cross_Entropy_Loss_Layer:: get_grad_wrt_predicted_label()
{
    return grad_wrt_predicted_label;
}
