//
//  Max_Pool.hpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include <iostream>
#include <armadillo>
#include <cassert>

using namespace arma;
using namespace std;

class Max_Pool
{
private:
    
    cube input;
    
    size_t i_H;
    size_t i_W;
    size_t i_D;
    
    
    size_t p_H;
    size_t p_W;
    size_t p_v_stride;
    size_t p_h_stride;
    
    cube output;
    cube grad_wrt_i;
    
public:
    
    Max_Pool(size_t i_H
             ,size_t i_W
             ,size_t i_D
             ,size_t p_H
             ,size_t p_W
             ,size_t p_v_stride
             ,size_t p_h_stride)
    {
        this->i_H = i_H;
        this->i_W = i_W;
        this->i_D = i_D;
        
        this->p_H = p_H;
        this->p_W = p_W;
        this->p_v_stride = p_v_stride;
        this->p_h_stride = p_h_stride;
    }
    
    void forward_pass(cube& input, cube& output);
    void backward_pass(cube& gradient);
    cube get_grad_wrt_i();
    
};
