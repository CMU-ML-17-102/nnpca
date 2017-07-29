//
//  regularizerOracle.h
//  Opt
//
//  Created by Sashan Jakkam Reddi on 2/18/16.
//  Copyright (c) 2016 Sashan Jakkam Reddi. All rights reserved.
//

#ifndef Opt_regularizerOracle_h
#define Opt_regularizerOracle_h

#include "Vector.h"
#include <cmath>

void computeRegGradient(Vector &g, const Vector &x, double alpha, double lambda){
    
    int d = x.size();
    
    for (int i=0; i < d; i++) {
        g[i] = (2*lambda*alpha*x[i])/(pow((1 + alpha*pow(x[i],2)),2));
    }    
}

double computeRegFunctionGradient(const Vector &x, Vector &g, double lambda, double alpha){
    double fval = 0.0;
    int d = x.size();
    
    for (int i=0; i < d; ++i){
        fval = fval + (pow(x[i],2)/(1 + alpha*pow(x[i],2)));
    }
    
    computeRegGradient(g, x, alpha, lambda);
    
    fval = fval * lambda * alpha;
    return fval;
}


#endif
