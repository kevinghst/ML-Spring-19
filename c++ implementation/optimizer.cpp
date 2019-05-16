//
// Created by Robbie on 5/3/2019.
//

#include "optimizer.h"

using namespace std;

Adam::Adam(double learning_rate, double b1, double b2) {
    this->learning_rate = learning_rate;
    this->b1 = b1;
    this->b2 = b2;
}
Adam::Adam() {
    Adam(0.0002, 0.5, 0.999);
}
xt::xarray<double> Adam::update(xt::xarray<double> *w, xt::xarray<double> *grad_wrt_w) {
    if (!this->initialized) {
        this->initialized = true;
        this->m = xt::zeros<double>(w->shape());
        this->v = xt::zeros<double>(grad_wrt_w->shape());
    }
    this->m = this->m * this->b1 + *grad_wrt_w * (1 - this->b1);
    this->v = this->v * this->b2 + xt::pow(*grad_wrt_w, 2) * (1 - this->b2);

    auto m_hat = this->m / (1 - this->b1);
    auto v_hat = this->v / (1 - this->b2);
    auto w_updt = (m_hat * this->learning_rate) / (xt::sqrt(v_hat) + this->eps);

    return *w - w_updt;
}

Adadelta::Adadelta( double rho, double eps) {
    this->rho = rho;
    this->eps = eps;
}
Adadelta::Adadelta() {
    Adadelta(0.95, 1.0e-6);
}
xt::xarray<double> Adadelta::update(xt::xarray<double> *w, xt::xarray<double> *grad_wrt_w) {
    if (!this->initialized) {
        this->initialized = true;
        this->w_updt = xt::zeros<double>(w->shape());
        this->E_w_updt = xt::zeros<double>(w->shape());
        this->E_grad = xt::zeros<double>(grad_wrt_w->shape());
    }

//    update average gradients at w
    this->E_grad = this->E_grad * this->rho + xt::pow(*grad_wrt_w, 2) * (1 - this->rho);

    auto RMS_delta_w = xt::sqrt(this->E_w_updt + this->eps);
    auto RMS_grad = xt::sqrt(this->E_grad + this->eps);

    auto adaptive_lr = RMS_delta_w / RMS_grad;

    this->w_updt = adaptive_lr * *grad_wrt_w;

    this->E_w_updt = this->E_w_updt * this->rho + xt::pow(this->w_updt, 2) * (1 - this->rho);

    return *w - this->w_updt;
}