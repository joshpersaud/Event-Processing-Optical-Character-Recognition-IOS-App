//
//  ReLU.hpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include <armadillo>

using namespace arma;
using namespace std;

class ReLU
{
private:
    cube input;
    cube output;
    
    size_t i_H;
    size_t i_W;
    size_t i_D;
    
    cube grad_wrt_i;
    
public:
    ReLU(size_t i_H
         ,size_t i_W
         ,size_t i_D)
    {
        this->i_H = i_H;
        this->i_W = i_W;
        this->i_D = i_D;
    }
    
    void forward_pass(cube& input, cube& output);
    void backward_pass(cube gradient);
    cube get_grad_wrt_i();
    
};
