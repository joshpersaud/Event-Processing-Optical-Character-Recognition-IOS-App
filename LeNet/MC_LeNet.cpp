//
//  LeNet.cpp
//  LeNet
//
//  Created by Daniel Vilajeti on 11/14/19.
//  Copyright © 2019 Daniel Vilajeti. All rights reserved.
//

#include "MC_LeNet.hpp"


void MC_LeNet:: train()
{
    
    cout << "TRAINING HAS BEGUN..." << endl;
    
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for( int batch = 0; batch < num_batches; batch++)
        {
            vec rand_batch(batch_size,fill::randu);
            rand_batch *= (t_set_len-1);
            
            for ( int i = 0; i < batch_size; i++)
            {
                c1->foward_pass(training_set[rand_batch[i]], c1_out);
                r1->forward_pass(c1_out, r1_out);
                p1->forward_pass(r1_out, p1_out);
                c2->foward_pass(p1_out, c2_out);
                r2->forward_pass(c2_out, r2_out);
                p2->forward_pass(r2_out, p2_out);
                n1->forward_pass(p2_out, n1_out);
                n1_out /= 100;
                s1->forward_pass(n1_out, s1_out);
                
                //cout << s1_out << endl;
                
                loss = e1->forward_pass(training_labels[rand_batch[i]], s1_out);
                //cout << "Error for the batch: " << loss << endl << endl;
                
                cum_loss += loss;
                
                e1->backward_pass();
                
                vec grad_wrt_predicted_label = e1->get_grad_wrt_predicted_label();
                s1->backward_pass(grad_wrt_predicted_label);
                vec grad_wrt_s1_i = s1->get_grad_wrt_i();
                n1->backward_pass(grad_wrt_s1_i);
                cube grad_wrt_n1_i = n1->get_grad_wrt_i();
                p2->backward_pass(grad_wrt_n1_i);
                cube grad_wrt_p2_i = p2->get_grad_wrt_i();
                r2->backward_pass(grad_wrt_p2_i);
                cube grad_wrt_r2_i = r2->get_grad_wrt_i();
                c2->backward_pass(grad_wrt_r2_i);
                cube grad_wrt_c2_i = c2->get_grad_wrt_i();
                p1->backward_pass(grad_wrt_c2_i);
                cube grad_wrt_p1_i = p1->get_grad_wrt_i();
                r1->backward_pass(grad_wrt_p1_i);
                cube grad_wrt_r1_i = r1->get_grad_wrt_i();
                c1->backward_pass(grad_wrt_r1_i);
            }
            n1->update_weights_and_biases(batch_size, learning_rate);
            c1->update_filters(batch_size, learning_rate);
            c2->update_filters(batch_size, learning_rate);

        }
        
        cout << "PERCENT COMPLETE: " << ((epoch+1.0)/epochs) * 100.0 << "%" << endl;
        cout << "EPOCH #" << epoch+1 << " loss: " << cum_loss/(batch_size*num_batches) << endl;
        
        cum_loss = 0.0;
    }
    
    this->convolution1_filters = c1->get_filters();
    this->convolution2_filters = c2->get_filters();
    this->network_biases = n1->get_biases();
    this->network_weights = n1->get_weights();
    
    write_training_results_to_file("/Users/danielvilajeti/Documents/COMPUTER_SCIENCE/CAPSTONE/LeNet/LeNeT/CNN.txt");
    
    cout << "TRAINING COMPLETE!" << endl;
}
vec MC_LeNet:: predict(cube img)
{
    c1->foward_pass(img, c1_out);
    r1->forward_pass(c1_out, r1_out);
    p1->forward_pass(r1_out, p1_out);
    c2->foward_pass(p1_out, c2_out);
    r2->forward_pass(c2_out, r2_out);
    p2->forward_pass(r2_out, p2_out);
    n1->forward_pass(p2_out, n1_out);
    n1_out /= 100;
    s1->forward_pass(n1_out, s1_out);
    
    return s1_out;
}

void MC_LeNet:: write_training_results_to_file(string file_path)
{
    ofstream results(file_path);
    
    for(int i = 0; i < convolution1_filters.size(); i++)
    {
        for(int j = 0; j < convolution1_filters[i].n_rows; j++)
        {
            for(int k = 0; k < convolution1_filters[i].n_cols; k++)
            {
                results << convolution1_filters[i].row(j)(k) << " ";
            }
            
            results << "\n";
        }
    }
    
    results << "\n";
    
    for(int i = 0; i < convolution2_filters.size(); i++)
    {
        for(int j = 0; j < convolution2_filters[i].n_rows; j++)
        {
            for(int k = 0; k < convolution2_filters[i].n_cols; k++)
            {
                results << convolution2_filters[i].row(j)(k) << " ";
            }
            
            results << "\n";
        }
    }
    
    results << "\n";
    
    for(int i = 0; i < network_weights.n_rows; i++)
    {
        for(int j = 0; j < network_weights.n_cols; j++)
        {
            results << network_weights.row(i)(j) << " ";
        }
        
        results << "\n";
    }
    
    results << "\n";
    
    for (int i = 0; i < network_biases.n_elem; i++)
    {
        results << network_biases[i] << " ";
    }
    
    results << "\n";
    results << "--------------------------------";
    results << "\n";
    
    results.close();
}


