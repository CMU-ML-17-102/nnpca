//
//  nnpca_solvers.h
//  Opt
//
//  Created by Sashan Jakkam Reddi on 5/6/16.
//  Copyright (c) 2016 Sashan Jakkam Reddi. All rights reserved.
//

#ifndef Opt_nnpca_solvers_h
#define Opt_nnpca_solvers_h

#include <iostream>
#include <random>
#include <vector>
#include "Vector.h"
#include "pcaOracle.h"
#include <stdio.h>


void nnproject(Vector &x, const int num_features){
    double prod = 0.0;
    for (int i = 0; i < num_features; ++i) {
        if (x[i] <= 0) {
            x[i] = 0.0;
            //prod = prod + (x[i]*x[i]);
        } else {
            prod = prod + (x[i]*x[i]);
        }
    }

    
    if (prod > 1.0) {
        for (int i = 0; i < num_features; ++i) {
            x[i] = x[i]/std::sqrt(prod);
        }
    }
    
    
    /*
    double tol = 0.00001;
    prod = 0.0;
    for (int i = 0; i < num_features; ++i) {
        if (x[i] < 0) {
            std::cout << "danger" << std::endl;
        }
        
        prod = prod + x[i]*x[i];
    }
    if (prod > 1.0+tol) {
        std::cout << "danger" << std::endl;
    }
     */
}


void initialize(std::vector<double> &A,
                Vector &Ag,
                Vector &x,
                const int num_examples,
                const int num_features,
                const std::vector<SparseVec> &examples,
                const double lambda,
                const double alpha,
                const double step){
    
    for(int i = 0; i < num_examples; ++i) {
        double lg;
        computeGradientP(lg, x, examples[i]);
        
        A[i] = lg;
        VectorIterator<SparseVec>  sparse_iterator(examples[i]);
        for(; sparse_iterator; sparse_iterator.next()) {
            int k = sparse_iterator.index();
            double val = sparse_iterator.value();
            x[k] = x[k] - step*(lg*val);
            Ag[k] = Ag[k] + (lg*val)/num_examples;
        }
        nnproject(x, num_features);
    }
}

void sgdSolver(Vector &x0,
               const int num_examples,
               const int num_features,
               const std::vector<SparseVec> &examples,
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
    initialize(A, Ag, x0, num_examples, num_features, examples, lambda, alpha, step);
    
    for (std::vector<double>::iterator it_eta = eta0.begin() ; it_eta != eta0.end(); ++it_eta){
        for (std::vector<double>::iterator it_b = b.begin() ; it_b != b.end(); ++it_b){
            
            int num_ifo = 0;
            
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
                    objective = computeFunctionGradient(x,tlg, examples);
                    
                    sq_gradient = 0.0;
                    for (int i =0; i < num_features;++i){
                        sq_gradient = sq_gradient + pow(tlg[i],2);
                    }
                    
                    LOG("update=" << i << " ifo=" << num_ifo << " eta0=" << (*it_eta) << " b=" << (*it_b) << " obj=" << objective << " sqg=" << sq_gradient);
                    
                    FILE* pFile = fopen(outputfile.c_str(), "a");
                    fprintf(pFile, "%i \t %i \t %.20f \t %.20f \t %.20f \t %.20f\n",i,num_ifo,(*it_eta),(*it_b),objective,sq_gradient);
                    fclose(pFile);
                }
                
                Vector lg(num_features);
                Vector rg(num_features);
                computeGradient(lg, x, examples[j]);
                num_ifo++;
                step = (*it_eta)/(1 + (*it_b)*(i/num_examples));
                
                for (int k=0; k < num_features; ++k){
                    x[k] = x[k] - step*(lg[k]);
                }
                nnproject(x,num_features);
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
    initialize(A, Ag, x0, num_examples, num_features, examples, lambda, alpha, step);
    
    for (std::vector<double>::iterator it_eta = eta0.begin() ; it_eta != eta0.end(); ++it_eta){
        step = (*it_eta);
        
        int num_ifo = 0;
        Vector x(num_features);
        for (int i=0; i < num_features; ++i){
            x[i] = x0[i];
        }
        
        for(int i = 0; i < num_updates; ++i) {
            int j = u(r);
            
            if ((i % 10000) == 0){
                Vector tlg(num_features);
                objective = computeFunctionGradient(x,tlg, examples);
                
                sq_gradient = 0.0;
                for (int i =0; i < num_features;++i){
                    sq_gradient = sq_gradient + pow(tlg[i],2);
                }
                
                LOG("update=" << i << " ifo=" << num_ifo << " eta0=" << (*it_eta) << " obj=" << objective << " sqg=" << sq_gradient);
                
                FILE* pFile = fopen(outputfile.c_str(), "a");
                fprintf(pFile, "%i \t %i \t %.20f \t %.20f \t %.20f\n",i,num_ifo,(*it_eta),objective,sq_gradient);
                fclose(pFile);
            }
            
            double lg;
            computeGradientP(lg, x, examples[j]);
            num_ifo++;
            
            for (int k=0; k < num_features; ++k){
                x[k] = x[k] - step*(Ag[k]);
            }
            
            VectorIterator<SparseVec>  sparse_iterator(examples[j]);
            for(; sparse_iterator; sparse_iterator.next()) {
                int k = sparse_iterator.index();
                double val = sparse_iterator.value();
                x[k] = x[k] - step*(lg*val - A[j]*val);
                Ag[k] = Ag[k] + (lg*val - A[j]*val)/num_examples;
            }
            A[j] = lg;
            nnproject(x, num_features);
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

void svrgSolver(Vector &x0,
                const int num_examples,
                const int num_features,
                const std::vector<SparseVec> &examples,
                const double lambda,
                const double alpha,
                double step,
                std::vector<double> eta0,
                const int num_updates,
                std::string outputfile){
    
    LOG(" Running SVRG");
    std::default_random_engine r;
    std::uniform_int_distribution<int> u(0, num_examples-1);
    double objective = 0.0;
    double best_objective = 10000000.0;
    double best_config_eta = 0.0;
    double sq_gradient = 0.0;
    
    std::vector<double> A(num_examples);
    Vector Ag(num_features);
    
    /* Pass Through initialization */
    initialize(A, Ag, x0, num_examples, num_features, examples, lambda, alpha, step);
    
    for (std::vector<double>::iterator it_eta = eta0.begin() ; it_eta != eta0.end(); ++it_eta){
        step = (*it_eta);
        
        Vector gsnapshot(num_features);
        Vector xsnapshot(num_features);
        int epoch_size = num_examples;
        int num_ifo = 0;
        
        Vector x(num_features);
        for (int i=0; i < num_features; ++i){
            x[i] = x0[i];
        }
        
        for(int i = 0; i < num_updates; ++i) {
            int j = u(r);
            
            if ((i % 10000) == 0){
                Vector tlg(num_features);
                objective = computeFunctionGradient(x,tlg, examples);
                
                sq_gradient = 0.0;
                for (int i =0; i < num_features;++i){
                    sq_gradient = sq_gradient + pow(tlg[i],2);
                }
                
                LOG("update=" << i << " ifo=" << num_ifo << " eta0=" << (*it_eta) << " obj=" << objective << " sqg=" << sq_gradient);
                
                FILE* pFile = fopen(outputfile.c_str(), "a");
                fprintf(pFile, "%i \t %i \t %.20f \t %.20f \t %.20f\n",i,num_ifo,(*it_eta),objective,sq_gradient);
                fclose(pFile);
            }
            
            
            if ((i % epoch_size) == 0) {
                // end of epoch
                Vector fg(num_features);
                objective = computeFunctionGradient(x,fg, examples);
                gsnapshot = fg;
                xsnapshot = x;
                num_ifo = num_ifo + num_examples;
            }
            
            double lg,lgsnapshot;
            computeGradientP(lg, x, examples[j]);
            computeGradientP(lgsnapshot, xsnapshot, examples[j]);
            num_ifo = num_ifo + 2;
            
            for (int k=0; k < num_features; ++k){
                x[k] = x[k] - step*(gsnapshot[k]);
            }
            
            VectorIterator<SparseVec>  sparse_iterator(examples[j]);
            for(; sparse_iterator; sparse_iterator.next()) {
                int k = sparse_iterator.index();
                double val = sparse_iterator.value();
                x[k] = x[k] - step*(lg*val - lgsnapshot*val);
            }
            nnproject(x, num_features);
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


void fsvrgSolver(Vector &x0,
                const int num_examples,
                const int num_features,
                const std::vector<SparseVec> &examples,
                const double lambda,
                const double alpha,
                double step,
                std::vector<double> eta0,
                const int num_updates,
                std::string outputfile){
    
    LOG(" Running SVRG");
    std::default_random_engine r;
    std::uniform_int_distribution<int> u(0, num_examples-1);
    double objective = 0.0;
    double best_objective = 10000000.0;
    double best_config_eta = 0.0;
    double sq_gradient = 0.0;
    
    std::vector<double> A(num_examples);
    Vector Ag(num_features);
    
    /* Pass Through initialization */
    initialize(A, Ag, x0, num_examples, num_features, examples, lambda, alpha, step);
    

    for (std::vector<double>::iterator it_eta = eta0.begin() ; it_eta != eta0.end(); ++it_eta){
        step = (*it_eta);
        
        
        Vector gsnapshot(num_features);
        Vector gisnapshot(num_examples);
        int epoch_size = num_examples;
        int num_ifo = 0;
        Vector x(num_features);
        
        for (int i=0; i < num_features; ++i){
            x[i] = x0[i];
        }
        
        for(int i = 0; i < num_updates; ++i) {
            int j = u(r);
            
            if ((i % 10000) == 0){
                Vector tlg(num_features);
                objective = computeFunctionGradient(x,tlg, examples);
                
                sq_gradient = 0.0;
                for (int i =0; i < num_features;++i){
                    sq_gradient = sq_gradient + pow(tlg[i],2);
                }
                
                LOG("update=" << i << " ifo=" << num_ifo << " eta0=" << (*it_eta) << " obj=" << objective << " sqg=" << sq_gradient);
                
                FILE* pFile = fopen(outputfile.c_str(), "a");
                fprintf(pFile, "%i \t %i \t %.20f \t %.20f \t %.20f\n",i,num_ifo,(*it_eta),objective,sq_gradient);
                fclose(pFile);
            }
            
            if ((i % epoch_size) == 0) {
                // end of epoch
                Vector fg(num_features);
                Vector gi(num_examples);
                objective = computeFullGradient(x,gi,fg, examples);
                gsnapshot = fg;
                gisnapshot = gi;
                num_ifo = num_ifo + num_examples;
            }
            
            double lg,lgsnapshot;
            computeGradientP(lg, x, examples[j]);
            lgsnapshot = gisnapshot[j];
            num_ifo = num_ifo + 1;
            
            for (int k=0; k < num_features; ++k){
                x[k] = x[k] - step*(gsnapshot[k]);
            }
            
            VectorIterator<SparseVec>  sparse_iterator(examples[j]);
            for(; sparse_iterator; sparse_iterator.next()) {
                int k = sparse_iterator.index();
                double val = sparse_iterator.value();
                x[k] = x[k] - step*(lg*val - lgsnapshot*val);
            }
            nnproject(x, num_features);
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
