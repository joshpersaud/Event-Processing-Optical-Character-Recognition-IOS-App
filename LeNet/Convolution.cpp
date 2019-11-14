//
//  Convolution.cpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include "Convolution.hpp"


void Convolution:: foward_pass(cube& input, cube& output)
{

    output = zeros(i_H - (f_H - 1)
                   ,i_W - (f_W - 1)
                   ,num_filters);
    
    for ( int i = 0; i < num_filters; i++)
    {
        for ( int j = 0; j < i_H; j++)
        {
            for ( int k = 0; k < i_W; k++)
            {
                //Get dot product between filter and image
                output(j,k,i) = dot(vectorise(input.subcube(i
                                                            ,j
                                                            ,0
                                                            ,i + (f_H - 1)
                                                            ,j + (f_W - 1)
                                                            ,i_D-1))
                                    ,vectorise(filters[i]));
            }
        }
    }
    
    this->input = input;
    this->output = output;
    
}
void Convolution:: backward_pass(cube& gradient)
{
    grad_i = zeros(i_H,i_W,i_D);
    
    //Calculate gradient with respect to input
    for (int i = 0; i < num_filters; i++)
    {
        for (int j = 0; j <output.n_rows; j++)
        {
            for(int k = 0; k < output.n_cols; k++)
            {
                cube temp;
                temp = zeros(i_H,i_W,i_D);
                temp.subcube(j
                             ,k
                             ,0
                             ,j + (f_H - 1)
                             ,k + (f_W - 1)
                             ,i_D - 1) = filters[i];
                grad_i += gradient.slice(i)(j,k) * temp;
            }
            
        }
    }
    
    accum_grad_i += grad_i;
    
    grad_f.clear();
    grad_f.resize(num_filters);
    for(int i = 0; i < num_filters; i++)
        grad_f[i] = zeros(f_H,f_W,i_D);
    
    for (int i = 0; i < num_filters; i++)
    {
        for (int j = 0; j <output.n_rows; j++)
        {
            for(int k = 0; k < output.n_cols; k++)
            {
                cube temp;
                temp = zeros(f_H,f_W,i_D);
                temp.subcube(j
                             ,k
                             ,0
                             ,j + (f_H - 1)
                             ,k + (f_W - 1)
                             ,i_D - 1) = filters[i];
                grad_f[i] += gradient.slice(i)(j,k) * temp;
            }
            
        }
    }
    
    for ( int i = 0; i < num_filters; i++)
        accum_grad_f[i] += grad_f[i];
    
}
void Convolution:: update_filters(size_t batch_size, double learning_rate)
{
    for(int i = 0; i < batch_size; i++)
    {
        filters[i] -= learning_rate * (accum_grad_f[i]/batch_size);
    }
    
    reset_accum_grad();
}

vector<cube> Convolution:: get_filters()
{
    return filters;
}

cube Convolution:: get_grad_wrt_i()
{
    return grad_i;
}

vector<cube> Convolution:: get_grad_wrt_f()
{
    return grad_f;
}
