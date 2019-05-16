//
// Created by Robbie on 5/3/2019.
//

#include "loss_function.h"


xt::xarray<double> SquareLoss::loss(xt::xarray<double> *y, xt::xarray<double> *y_pred) {
    return xt::pow(*y - *y_pred, 2) * 0.5;
}
xt::xarray<double> SquareLoss::gradient(xt::xarray<double>* y, xt::xarray<double>* y_pred) {
    return -(*y - *y_pred);
}
xt::xarray<double> SquareLoss::acc(xt::xarray<double>* y, xt::xarray<double>* y_pred) {
    return 0;
//    return xt::xarray<double> (1,0);
}
xt::xarray<double> CrossEntropy::loss(xt::xarray<double>* y, xt::xarray<double>* y_pred) {
    auto p = xt::clip(*y_pred, 1.0e-15, 1.0 - 1.0e-15);
    return -1 * *y * xt::log(p) - (1 - *y) * xt::log(1.0 - p);
}
xt::xarray<double> CrossEntropy::gradient(xt::xarray<double>* y, xt::xarray<double>* y_pred) {
    return (*y - *y_pred) * -1.0;
}
xt::xarray<double> CrossEntropy::acc(xt::xarray<double>* y, xt::xarray<double>* y_pred) {
    auto y_true = xt::argmax(*y, 1);
    auto y_p = xt::argmax(*y_pred,1);
//    return xt::sum(y_true == y_p, 0) / xt::prod(y_true.shape());
    return 0;
}