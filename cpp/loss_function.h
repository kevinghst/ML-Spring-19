//
// Created by Robbie on 5/3/2019.
//

#ifndef ML_SPRING19_PROJECT_LOSS_FUNCTION_H
#define ML_SPRING19_PROJECT_LOSS_FUNCTION_H

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>

using namespace std;

class Loss {
public:
    virtual xt::xarray<double> loss(xt::xarray<double> *, xt::xarray<double> *)=0;
    virtual xt::xarray<double> gradient(xt::xarray<double> *, xt::xarray<double> *)=0;
    virtual xt::xarray<double> acc(xt::xarray<double> *, xt::xarray<double> *)=0;
};
class SquareLoss: public Loss {
public:
    xt::xarray<double> loss(xt::xarray<double> *, xt::xarray<double> *) override;
    xt::xarray<double> gradient(xt::xarray<double> *, xt::xarray<double> *) override;
    xt::xarray<double> acc(xt::xarray<double> *, xt::xarray<double> *) override;
};
class CrossEntropy: public Loss {
public:
    xt::xarray<double> loss(xt::xarray<double> *, xt::xarray<double> *) override;
    xt::xarray<double> gradient(xt::xarray<double> *, xt::xarray<double> *) override;
    xt::xarray<double> acc(xt::xarray<double> *, xt::xarray<double> *) override;
};

#endif //ML_SPRING19_PROJECT_LOSS_FUNCTION_H
