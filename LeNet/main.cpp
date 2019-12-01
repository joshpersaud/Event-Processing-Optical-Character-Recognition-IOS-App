//
//  main.cpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/13/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//
#include "MC_LeNet.hpp"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <armadillo>

using namespace arma;
using namespace std;

#define I_H_c1 28
#define I_W_c1 28
#define I_D_c1 1

#define F_H 5
#define F_W 5
#define STRIDE_H 1
#define STRIDE_V 1
#define NUM_FILT 6

#define P_H 2
#define P_W 2
#define P_H_STRIDE 2
#define P_V_STRIDE 2

#define LEARNING_RATE .05
#define EPOCHS 10
#define BATCH_SIZE 10

unordered_map<int, int> load_training_set(mat& train_data,string path_to_set);
void get_training_imgs(vector<cube>& training_imgs,mat train_data);
void get_training_labels(vector<vec>& training_labels,mat train_data,unordered_map<int, int>map);

int main() {
    
    string train_path = "/Users/danielvilajeti/Documents/COMPUTER_SCIENCE/CAPSTONE/LeNet/LeNeT/training_set.txt";
    mat trainData;
    
    unordered_map<int, int> map;
    vector<cube> training_imgs;
    vector<vec> training_labels;
    
    map = load_training_set(trainData, train_path);
    get_training_imgs(training_imgs, trainData);
    get_training_labels(training_labels, trainData, map);
    
    assert(training_imgs.size() == training_labels.size());
    
    MC_LeNet net(training_imgs,training_labels,I_H_c1
                                                ,I_W_c1
                                                ,I_D_c1
                                                ,F_H
                                                ,F_W
                                                ,NUM_FILT
                                                ,STRIDE_H
                                                ,STRIDE_V
                                                ,P_H
                                                ,P_W
                                                ,P_H_STRIDE
                                                ,P_V_STRIDE
                                                ,map.size()
                                                ,BATCH_SIZE
                                                ,EPOCHS
                                                ,LEARNING_RATE);
    net.train();
    
    return 0;
}

unordered_map<int, int> load_training_set(mat& train_data,string path_to_set)
{
    train_data.load(path_to_set,raw_ascii);
    vector<int> labels;
    unordered_map<int, int> labels_map;
    
    for (size_t i = 0; i < train_data.n_rows; i++)
    {
        labels_map[train_data.row(i)[0]]++;
    }
    
    for(auto i : labels_map)
    {
        labels.push_back(i.first);
    }
    
    sort(labels.begin(),labels.end());
    
    for (int i = 0; i < labels.size(); i++)
    {
        labels_map[labels[i]] = i;
    }
    
    return labels_map;
}

void get_training_imgs(vector<cube>& training_imgs,mat train_data)
{
    for(size_t i = 0; i < train_data.n_rows; i++)
    {
        cube img(I_H_c1,I_W_c1,1,fill::zeros);
        
        for(size_t j = 0; j < I_H_c1; j++)
        {
            img.slice(0).row(j) = train_data.row(i).subvec(I_H_c1*j+1, I_W_c1*j+I_W_c1);
        }
        
        img.slice(0) = arma::normalise(img.slice(0));
        training_imgs.push_back(img);
    }
}
void get_training_labels(vector<vec>& training_labels,mat train_data,unordered_map<int, int>map)
{
    for(size_t i = 0; i < train_data.n_rows; i++)
    {
        vec label(map.size(),fill::zeros);
        
        label[map[train_data.row(i)(0)]] = 1;
        
        training_labels.push_back(label);
    }
}
