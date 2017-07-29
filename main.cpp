//
//  main.cpp
//  Opt
//
//  Created by Sashank Jakkam Reddi on 2/18/16.
//  Copyright (c) 2016 Sashan Jakkam Reddi. All rights reserved.
//

#include <iostream>
#include "DataReader.h"
#include "nnpca_solvers.h"

int main(int argc, const char * argv[]) {

    int num_features;
    std::vector<SparseVec> examples;
    std::vector<double> labels;
    std::vector<SparseVec> test_examples;
    std::vector<double> test_labels;
    // input file here
    std::string training_file = "/Users/sjakkamr/Documents/workspace/SAGA/Opt/Opt/data/rcv1.bin";
    
    bool normalize_examples = true;

    BinaryDataReader::readTrainingFile(training_file.c_str(), normalize_examples, examples, labels, num_features);
    
    // Split into Training and Test data
    std::default_random_engine r(0);
    std::uniform_int_distribution<int> u(1, 100);
    
    int end = examples.size()-1; int idx = 0;
    while(idx < end) {
        int p = u(r);
        if (p <= 20) {
            test_examples.push_back(examples[idx]);
            test_labels.push_back(labels[idx]);
            examples[idx] = examples[end];
            labels[idx] = labels[end];
            --end;
        } else {++idx;}
    }
    
    examples.erase(examples.begin()+end, examples.end());
    labels.erase(labels.begin()+end, labels.end());
    
    /////////////////////////////////////////
    
    int num_examples = examples.size();

    double step = 1;
    int num_updates = 100*num_examples;
    double lambda = 0.001;
    double alpha = 1;
    
    for (int i=0;i < num_examples; ++i){
        if (labels[i] == 0){
            labels[i] = -1;
        }
    }
    
    for (int i=0;i < test_examples.size(); ++i){
        if (test_labels[i] == 0){
            test_labels[i] = -1;
        }
    }
    
    double init = 1.0/sqrt(num_features);
    Vector x0(num_features);
    for (int i = 0 ; i < num_features; ++i) {
        x0[i] = init;
    }
    
    /* Configurations */
    std::vector<double> eta0;
    std::vector<double> b;
    eta0.push_back(0.1);eta0.push_back(1);eta0.push_back(10);
    b.push_back(0);b.push_back(0.1);b.push_back(1);b.push_back(5);
    sgdSolver(x0, num_examples, num_features, examples, lambda, alpha, step, eta0,b, num_updates,"/Users/sjakkamr/Documents/workspace/SAGA/Opt/Opt/nips_results/sgdrcv1.txt");
    
    Vector x1(num_features);
    for (int i = 0 ; i < num_features; ++i) {
        x1[i] = init;
    }

    /* Configurations */
    std::vector<double> eta1;
    eta1.push_back(0.1);eta1.push_back(0.5);eta1.push_back(1);eta1.push_back(2);eta1.push_back(4);
    sagaSolver(x1, num_examples, num_features, examples, lambda, alpha, step, eta1, num_updates,"/Users/sjakkamr/Documents/workspace/SAGA/Opt/Opt/nips_results/sagarcv1.txt");
    
    ///////////////////////////////
    
    Vector x2(num_features);
    for (int i = 0 ; i < num_features; ++i) {
        x2[i] = init;
    }
    /* Configurations */
    std::vector<double> eta2;
    eta2.push_back(0.1);eta2.push_back(0.5);eta2.push_back(1);eta2.push_back(2);eta2.push_back(4);
    svrgSolver(x2, num_examples, num_features, examples, lambda, alpha, step, eta2, num_updates,"/Users/sjakkamr/Documents/workspace/SAGA/Opt/Opt/nips_results/svrgrcv1.txt");
    
    /////////////////////////////////
    
    Vector x3(num_features);
    for (int i = 0 ; i < num_features; ++i) {
        x3[i] = init;
    }
    /* Configurations */
    std::vector<double> eta3;
    eta3.push_back(0.1);eta3.push_back(0.5);eta3.push_back(1);eta3.push_back(2);eta3.push_back(4);
    fsvrgSolver(x3, num_examples, num_features, examples, lambda, alpha, step, eta3, num_updates,"/Users/sjakkamr/Documents/workspace/SAGA/Opt/Opt/nips_results/fsvrgrcv1.txt");
    

    /*
    
    Vector x0(num_features);
    
    // Configurations
    std::vector<double> eta0;
    std::vector<double> b;
    eta0.push_back(0.1);eta0.push_back(1);eta0.push_back(10);eta0.push_back(100);
    b.push_back(0);b.push_back(0.1);b.push_back(1);b.push_back(5);b.push_back(10);
    
    //std::vector<double> eta0(2);
    //std::vector<double> b(1);
    //eta0.push_back(0.1);eta0.push_back(1);
    //b.push_back(0);
    
    sgdSolver(x0, num_examples, num_features, examples, labels, test_examples, test_labels, lambda, alpha, step, eta0,b, num_updates,"/Users/sjakkamr/Documents/workspace/SAGA/Opt/Opt/results/sgdrcv1.txt");
    
    Vector x1(num_features);
    
    // Configurations
    std::vector<double> eta1;
    eta1.push_back(0.1);eta1.push_back(0.5);eta1.push_back(1);eta1.push_back(2);eta1.push_back(4);
    
    //std::vector<double> eta1(2);
    //eta1.push_back(0.1);eta1.push_back(0.5);
    //eta1.push_back(4);
    sagaSolver(x1, num_examples, num_features, examples, labels, test_examples, test_labels, lambda, alpha, step, eta1, num_updates,"/Users/sjakkamr/Documents/workspace/SAGA/Opt/Opt/results/sagarcv1.txt");
     */
    
    return 0;
}
