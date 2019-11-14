//
//  Soft_Max.hpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include <iostream>
#include <armadillo>
using namespace arma;
using namespace std;

class Soft_Max
{
private:
    
    size_t numInputs;
    vec input;
    vec output;
    
    vec grad_wrt_i;
    
public:
    
    Soft_Max(size_t numInputs)
    {
        this->numInputs = numInputs;
    }
    
    void forward_pass(vec& input,vec& output);
    void backward_pass(vec& gradient);
    vec get_grad_wrt_i();
    
};
