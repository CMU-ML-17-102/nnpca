//
//  logisticOracle.h
//  Opt
//
//  Created by Sashan Jakkam Reddi on 2/18/16.
//  Copyright (c) 2016 Sashan Jakkam Reddi. All rights reserved.
//

#ifndef Opt_logisticOracle_h
#define Opt_logisticOracle_h

#include "Vector.h"


void computeGradient(Vector &g, const Vector &x, const SparseVec example, const double label){
    double prod = 0.0;
    
    VectorIterator<SparseVec>  sparse_iterator(example);
    
    for(; sparse_iterator; sparse_iterator.next()) {
        prod += sparse_iterator.value() * x[sparse_iterator.index()];
    }
    
    VectorIterator<SparseVec> sparse_iter(example);
    for(; sparse_iter; sparse_iter.next()) {
        int idx = sparse_iter.index();
        double val = sparse_iter.value();
        g[idx] = -(1.0 - (1.0/(1.0 + exp(-label*prod))))*label*val;
    }
}

void computeGradientP(double &g, const Vector &x, const SparseVec example, const double label){
    double prod = 0.0;
    
    VectorIterator<SparseVec>  sparse_iterator(example);
    
    for(; sparse_iterator; sparse_iterator.next()) {
        prod += sparse_iterator.value() * x[sparse_iterator.index()];
    }
    g = -(1.0 - (1.0/(1.0 + exp(-label*prod))))*label;
}

void addGradient(Vector &g, const Vector &x, const SparseVec example, const double label){
    double prod = 0.0;
    
    VectorIterator<SparseVec>  sparse_iterator(example);
    
    for(; sparse_iterator; sparse_iterator.next()) {
        prod += sparse_iterator.value() * x[sparse_iterator.index()];
    }
    
    VectorIterator<SparseVec> sparse_iter(example);
    for(; sparse_iter; sparse_iter.next()) {
        int idx = sparse_iter.index();
        double val = sparse_iter.value();
        g[idx] = g[idx] -(1.0 - (1.0/(1.0 + exp(-label*prod))))*label*val;
    }
}


double computeFunctionGradient(const Vector &x, Vector &g, const std::vector<SparseVec> examples, const std::vector<double> labels){
    double fval = 0.0;
    int num_examples = examples.size();
    int num_features = x.size();
    
    for (int i =0;i < num_examples; ++i){
        VectorIterator<SparseVec>  sparse_iterator(examples[i]);
        double prod = 0.0;
        for(; sparse_iterator; sparse_iterator.next()) {
            prod += sparse_iterator.value() * x[sparse_iterator.index()];
        }
        fval = fval + log(1.0 + exp(-labels[i]*prod));
        
        addGradient(g, x, examples[i], labels[i]);
    }
    
    for (int i = 0; i < num_features; ++i){
        g[i] = g[i]/num_examples;
    }
    
    fval = fval/num_examples;
    return fval;
}



#endif
