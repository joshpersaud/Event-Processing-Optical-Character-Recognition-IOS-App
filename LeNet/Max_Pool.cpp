//
//  Max_Pool.cpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include "Max_Pool.hpp"

void Max_Pool:: forward_pass(cube& input, cube& output)
{
    output = zeros(i_H/p_H,i_W/p_W,i_D);
    
    for (int i = 0; i < i_D; i++)
    {
        for (int j = 0; j < i_H - p_H; j += p_v_stride)
        {
            for (int k = 0; k < i_W - p_W; k += p_h_stride)
            {
                output.slice(i)(j/p_v_stride,k/p_h_stride) = input.slice(i).submat(j
                                                                                   ,k
                                                                                   ,j + (p_H - 1)
                                                                                   ,k + (p_W - 1)).max();
            }
        }
    }
    
    this->input = input;
    this->output = output;
}
void Max_Pool:: backward_pass(cube& gradient)
{
    grad_wrt_i = zeros(i_H,i_W,i_D);
    
    for (int i = 0; i < i_D; i++)
    {
        for (int j = 0; j + p_H <= i_H; j++)
        {
            for (int k = 0; k + p_W <= i_W; k++)
            {
                mat tmp = zeros(p_H,p_W);
                tmp(input.slice(i).submat(j
                                          ,k
                                          ,j + (p_H - 1)
                                          ,k + (p_W - 1)).index_max()) = gradient.slice(i)(j/p_v_stride,k/p_h_stride);
                grad_wrt_i.slice(i).submat(j
                                           ,k
                                           ,j + (p_H - 1)
                                           ,k + (p_W - 1)) += tmp;
            }
        }
    }
}
cube Max_Pool:: get_grad_wrt_i()
{
    return grad_wrt_i;
}
