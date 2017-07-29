//
//  pcaOracle.h
//  Opt
//
//  Created by Sashan Jakkam Reddi on 5/7/16.
//  Copyright (c) 2016 Sashan Jakkam Reddi. All rights reserved.
//

#ifndef Opt_pcaOracle_h
#define Opt_pcaOracle_h


#include "Vector.h"


void computeGradient(Vector &g, const Vector &x, const SparseVec example){
    double prod = 0.0;
    
    VectorIterator<SparseVec>  sparse_iterator(example);
    
    for(; sparse_iterator; sparse_iterator.next()) {
        prod += sparse_iterator.value() * x[sparse_iterator.index()];
    }
    
    VectorIterator<SparseVec> sparse_iter(example);
    for(; sparse_iter; sparse_iter.next()) {
        int idx = sparse_iter.index();
        double val = sparse_iter.value();
        g[idx] = -prod*val;
    }
}

void computeGradientP(double &g, const Vector &x, const SparseVec example){
    double prod = 0.0;
    
    VectorIterator<SparseVec>  sparse_iterator(example);
    
    for(; sparse_iterator; sparse_iterator.next()) {
        prod += sparse_iterator.value() * x[sparse_iterator.index()];
    }
    g = -prod;
}

void addGradient(double &l, Vector &g, const Vector &x, const SparseVec example){
    double prod = 0.0;
    
    VectorIterator<SparseVec>  sparse_iterator(example);
    
    for(; sparse_iterator; sparse_iterator.next()) {
        prod += sparse_iterator.value() * x[sparse_iterator.index()];
    }
    l = -prod;
    VectorIterator<SparseVec> sparse_iter(example);
    for(; sparse_iter; sparse_iter.next()) {
        int idx = sparse_iter.index();
        double val = sparse_iter.value();
        g[idx] = g[idx] - prod*val;
    }
}


double computeFunctionGradient(const Vector &x, Vector &g, const std::vector<SparseVec> examples){
    double fval = 0.0;
    int num_examples = examples.size();
    int num_features = x.size();
    
    for (int i =0;i < num_examples; ++i){
        VectorIterator<SparseVec>  sparse_iterator(examples[i]);
        double prod = 0.0;
        for(; sparse_iterator; sparse_iterator.next()) {
            prod += sparse_iterator.value() * x[sparse_iterator.index()];
        }
        fval = fval - 0.5*prod*prod;
        double l = 0.0;
        addGradient(l, g, x, examples[i]);
    }
    
    for (int i = 0; i < num_features; ++i){
        g[i] = g[i]/num_examples;
    }
    
    fval = fval/num_examples;
    return fval;
}

double computeFullGradient(const Vector &x, Vector &gi, Vector &g, const std::vector<SparseVec> examples){
    double fval = 0.0;
    int num_examples = examples.size();
    int num_features = x.size();
    
    for (int i =0;i < num_examples; ++i){
        VectorIterator<SparseVec>  sparse_iterator(examples[i]);
        double prod = 0.0;
        for(; sparse_iterator; sparse_iterator.next()) {
            prod += sparse_iterator.value() * x[sparse_iterator.index()];
        }
        fval = fval - 0.5*prod*prod;
        
        double l = 0.0;
        addGradient(l, g, x, examples[i]);
        gi[i] = l;
    }
    
    for (int i = 0; i < num_features; ++i){
        g[i] = g[i]/num_examples;
    }
    
    fval = fval/num_examples;
    return fval;
}

#endif
