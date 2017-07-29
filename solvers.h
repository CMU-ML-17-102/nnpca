//
//  solvers.h
//  Opt
//
//  Created by Sashan Jakkam Reddi on 2/18/16.
//  Copyright (c) 2016 Sashan Jakkam Reddi. All rights reserved.
//

#ifndef Opt_solvers_h
#define Opt_solvers_h

#include <iostream>
#include <random>
#include <vector>
#include "Vector.h"
#include "logisticOracle.h"
#include "regularizerOracle.h"
#include <stdio.h>


void initialize(std::vector<double> &A,
                Vector &Ag,
                Vector &x,
                const int num_examples,
                const int num_features,
                const std::vector<SparseVec> &examples,
                const std::vector<double> &labels,
                const double lambda,
                const double alpha,
                const double step){
    
    for(int i = 0; i < num_examples; ++i) {
        double lg;
        Vector rg(num_features);
        computeGradientP(lg, x, examples[i], labels[i]);
        computeRegGradient(rg,x, alpha, lambda);
        for (int k=0; k < num_features; ++k){
            x[k] = x[k] - step*(rg[k]);
        }
        
        A[i] = lg;
        VectorIterator<SparseVec>  sparse_iterator(examples[i]);
        for(; sparse_iterator; sparse_iterator.next()) {
            int k = sparse_iterator.index();
            double val = sparse_iterator.value();
            x[k] = x[k] - step*(lg*val);
            Ag[k] = Ag[k] + (lg*val)/num_examples;
        }
    }
}


double getClassificationError(Vector &x,
                            int num_features,
                            const std::vector<SparseVec> &test_examples,
                            const std::vector<double> &test_labels){
    int num_test = test_examples.size();
    double error = 0.0;
    for (int i=0; i < num_test; ++i){
        double inner_prod = 0.0;
        VectorIterator<SparseVec>  sparse_iterator(test_examples[i]);
        for(; sparse_iterator; sparse_iterator.next()) {
            int k = sparse_iterator.index();
            double val = sparse_iterator.value();
            inner_prod = inner_prod + x[k]*val;
        }
        double temp = test_labels[i];
        if (inner_prod*test_labels[i] < 0){
            error = error + 1.0;
        }
    }
    error  = error/num_test;
    return error;
}

void sgdSolver(Vector &x0,
               const int num_examples,
               const int num_features,
               const std::vector<SparseVec> &examples,
               const std::vector<double> &labels,
               const std::vector<SparseVec> &test_examples,
               const std::vector<double> &test_labels,
               const double lambda,
               const double alpha,
               double step,
               std::vector<double> eta0,
               std::vector<double> b,
               const int num_updates,
               std::string outputfile){
    
    LOG(" Running SGD ");
    std::default_random_engine r;
    std::uniform_int_distribution<int> u(0, num_examples-1);
    double objective = 0.0;
    double best_objective = 10000000.0;
    double best_config_eta = 0.0;
    double best_config_b = 0.0;
    double sq_gradient = 0.0;
    
    std::vector<double> A(num_examples);
    Vector Ag(num_features);
    
    /* Pass Through initialization */
    initialize(A, Ag, x0, num_examples, num_features, examples, labels, lambda, alpha, step);
    
    for (std::vector<double>::iterator it_eta = eta0.begin() ; it_eta != eta0.end(); ++it_eta){
        for (std::vector<double>::iterator it_b = b.begin() ; it_b != b.end(); ++it_b){
            
            Vector x(num_features);
            for (int i=0; i < num_features; ++i){
                x[i] = x0[i];
            }
            
            /* Run SGD algorithm */
            for(int i = 0; i < num_updates; ++i) {
                int j = u(r);
                
                if ((i % 10000) == 0){
                    Vector tlg(num_features);
                    Vector trg(num_features);
                    objective = computeFunctionGradient(x,tlg, examples, labels);
                    objective = objective + computeRegFunctionGradient(x, trg, lambda, alpha);
                    
                    sq_gradient = 0.0;
                    for (int i =0; i < num_features;++i){
                        sq_gradient = sq_gradient + pow(tlg[i] + trg[i],2);
                    }
                    
                    /// test error
                    double error = getClassificationError(x, num_features, test_examples, test_labels);
                    
                    LOG(i << " eta0=" << (*it_eta) << " b=" << (*it_b) << " obj=" << objective << " sq gradient=" << sq_gradient << " Test Error=" << error);
                    
                    FILE* pFile = fopen(outputfile.c_str(), "a");
                    fprintf(pFile, "%i \t %.20f \t %.20f \t %.20f \t %.20f \t %.20f\n",i,(*it_eta),(*it_b),objective,sq_gradient,error);
                    fclose(pFile);
                }
                
                Vector lg(num_features);
                Vector rg(num_features);
                computeGradient(lg, x, examples[j], labels[j]);
                computeRegGradient(rg,x, alpha, lambda);
                step = (*it_eta)/(1 + (*it_b)*(i/num_examples));
                
                for (int k=0; k < num_features; ++k){
                    x[k] = x[k] - step*(lg[k] + rg[k]);
                }
            }
            
            if (objective < best_objective){
                best_objective = objective;
                best_config_eta = *it_eta;
                best_config_b = *it_b;;
            }
        }
    }
    
    FILE* pFile = fopen(outputfile.c_str(), "a");
    fprintf(pFile, "%.20f \t %.20f \t %.20f \n",objective,best_config_eta,best_config_b);
    fclose(pFile);
}

void sagaSolver(Vector &x0,
                const int num_examples,
                const int num_features,
                const std::vector<SparseVec> &examples,
                const std::vector<double> &labels,
                const std::vector<SparseVec> &test_examples,
                const std::vector<double> &test_labels,
                const double lambda,
                const double alpha,
                double step,
                std::vector<double> eta0,
                const int num_updates,
                std::string outputfile){
    
    LOG(" Running SAGA");
    std::default_random_engine r;
    std::uniform_int_distribution<int> u(0, num_examples-1);
    double objective = 0.0;
    double best_objective = 10000000.0;
    double best_config_eta = 0.0;
    double sq_gradient = 0.0;
    
    std::vector<double> A(num_examples);
    Vector Ag(num_features);
    
    /* Pass Through initialization */
    initialize(A, Ag, x0, num_examples, num_features, examples, labels, lambda, alpha, step);
    
    for (std::vector<double>::iterator it_eta = eta0.begin() ; it_eta != eta0.end(); ++it_eta){
        step = (*it_eta);
        
        Vector x(num_features);
        for (int i=0; i < num_features; ++i){
            x[i] = x0[i];
        }
        
        for(int i = 0; i < num_updates; ++i) {
            int j = u(r);
            
            if ((i % 10000) == 0){
                Vector tlg(num_features);
                Vector trg(num_features);
                objective = computeFunctionGradient(x,tlg, examples, labels);
                objective = objective + computeRegFunctionGradient(x, trg, lambda, alpha);
                
                sq_gradient = 0.0;
                for (int i =0; i < num_features;++i){
                    sq_gradient = sq_gradient + pow(tlg[i] + trg[i],2);
                }
                
                /// test error
                double error = getClassificationError(x, num_features, test_examples, test_labels);
                
                LOG(i << " eta0=" << (*it_eta) << " obj=" << objective << " sq gradient=" << sq_gradient << " Test Error=" << error);
                
                FILE* pFile = fopen(outputfile.c_str(), "a");
                fprintf(pFile, "%i \t %.20f \t %.20f \t %.20f \t %.20f\n",i,(*it_eta),objective,sq_gradient, error);
                fclose(pFile);
            }
            
            double lg;
            Vector rg(num_features);
            computeGradientP(lg, x, examples[j], labels[j]);
            computeRegGradient(rg,x, alpha, lambda);
            
            
            for (int k=0; k < num_features; ++k){
                x[k] = x[k] - step*(rg[k] + Ag[k]);
            }
            
            VectorIterator<SparseVec>  sparse_iterator(examples[j]);
            for(; sparse_iterator; sparse_iterator.next()) {
                int k = sparse_iterator.index();
                double val = sparse_iterator.value();
                //x[k] = x[k] - step*(lg*val);
                x[k] = x[k] - step*(lg*val - A[j]*val);
                Ag[k] = Ag[k] + (lg*val - A[j]*val)/num_examples;
            }
            A[j] = lg;
        }
        
        if (objective < best_objective){
            best_objective = objective;
            best_config_eta = (*it_eta);
        }
    }
    
    FILE* pFile = fopen(outputfile.c_str(), "a");
    fprintf(pFile, "%.20f \t %.20f\n",objective,best_config_eta);
    fclose(pFile);
    
}

#endif
