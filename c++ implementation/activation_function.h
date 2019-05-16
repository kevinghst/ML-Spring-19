//
// Created by Robbie on 5/7/2019.
//

#ifndef ML_SPRING19_PROJECT_ACTIVATION_FUNCTION_H
#define ML_SPRING19_PROJECT_ACTIVATION_FUNCTION_H

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>

using namespace std;

class ActivationFunction {
public:
    virtual xt::xarray<double> function(xt::xarray <double> *x)=0;
    virtual xt::xarray<double> gradient(xt::xarray <double> *x)=0;
};
class Sigmoid : public ActivationFunction {
public:
    xt::xarray<double> function(xt::xarray <double> *x) override;
    xt::xarray<double> gradient(xt::xarray <double> *x) override;
};
class TanH : public ActivationFunction {
public:
    xt::xarray<double> function(xt::xarray <double> *x) override;
    xt::xarray<double> gradient(xt::xarray <double> *x) override;
};
class LeakyReLU : public ActivationFunction {
private:
    double alpha=0.2;
public:
//    LeakyReLU(double alpha);
    xt::xarray<double> function(xt::xarray <double> *x) override;
    xt::xarray<double> gradient(xt::xarray <double> *x) override;
};
class ReLU : public ActivationFunction {

public:
//    LeakyReLU(double alpha);
    xt::xarray<double> function(xt::xarray <double> *x) override;
    xt::xarray<double> gradient(xt::xarray <double> *x) override;
};

#endif //ML_SPRING19_PROJECT_ACTIVATION_FUNCTION_H
