//
// Created by Robbie on 5/7/2019.
//

#include "activation_function.h"

xt::xarray<double> Sigmoid::function(xt::xarray<double> *x) {
    return 1 / (1 + xt::exp(-*x));
}
xt::xarray<double> Sigmoid::gradient(xt::xarray<double> *x) {
    auto sig = this->function(x);
    return sig * (1 - sig);
}


xt::xarray<double> TanH::function(xt::xarray<double> *x) {
    return (2 / (1 + xt::exp(-2 * *x))) - 1;
}
xt::xarray<double> TanH::gradient(xt::xarray<double> *x) {
    auto tanh = this->function(x);
    return 1 - xt::pow(tanh, 2);
}


//LeakyReLU::LeakyReLU(double alpha) {
//    this->alpha = alpha;
//}
xt::xarray<double> LeakyReLU::function(xt::xarray<double> *x) {
    return xt::where(*x >= 0, *x, *x * this->alpha);
}
xt::xarray<double> LeakyReLU::gradient(xt::xarray<double> *x) {
    return xt::where(*x >= 0, 1, this->alpha);
}


xt::xarray<double> ReLU::function(xt::xarray<double> *x) {
    return xt::where(*x >= 0, *x, 0);
}
xt::xarray<double> ReLU::gradient(xt::xarray<double> *x) {
    return xt::where(*x >= 0, 1, 0);
}
