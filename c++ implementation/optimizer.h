//
// Created by Robbie on 5/3/2019.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xbuilder.hpp>

using namespace std;

class Optimizer {
public:
    virtual xt::xarray<double> update(xt::xarray<double> *w, xt::xarray<double> *grad_wrt_w)=0;
};

class Adam: public Optimizer {
    double eps = 1.0e-8;
    double learning_rate, b1, b2;
    bool initialized = false;
    xt::xarray<double> m,v;
public:
    Adam();
    Adam(double learing_rate, double b1, double b2);
    xt::xarray<double> update(xt::xarray<double> *w, xt::xarray<double> *grad_wrt_w) override ;
};
class Adadelta: public Optimizer {
    double eps, rho;
    bool initialized = false;
    xt::xarray<double> E_w_updt, E_grad, w_updt;
public:
    Adadelta();
    Adadelta(double rho, double eps);
    xt::xarray<double> update(xt::xarray<double> *w, xt::xarray<double> *grad_wrt_w) override ;
};


#endif //OPTIMIZER_H
