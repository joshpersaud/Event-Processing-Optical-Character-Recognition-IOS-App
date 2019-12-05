//
//  Cross_Entropy_Loss_Layer.hpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/30/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

class Cross_Entropy_Loss_Layer
{
private:
    size_t num_inputs;
    vec real_label;
    vec predicted_label;
    
    double loss;
    vec grad_wrt_predicted_label;
    
public:
    
    Cross_Entropy_Loss_Layer(size_t num_inputs)
    {
        this->num_inputs = num_inputs;
    }
    
    double forward_pass(vec& real_label, vec& predicted_label);
    void backward_pass();
    vec get_grad_wrt_predicted_label();
};
