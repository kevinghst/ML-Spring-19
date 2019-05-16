//
// Created by Robbie on 5/4/2019.
//

#ifndef ML_SPRING19_PROJECT_LAYER_H
#define ML_SPRING19_PROJECT_LAYER_H

#include <string>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include "optimizer.h"
#include "activation_function.h"

using namespace std;

typedef pair<int, int> Shape;

class Layer {
public:
    Shape input_shape;
    bool trainable;
    bool first_layer, latent_layer;
    void set_input_shape(Shape);
    xt::xarray<double> dot(xt::xarray<double> *a, xt::xarray<double> *b);
    virtual string layer_name()=0;
    virtual int parameters()=0;
    virtual void initialize(string optimizer)=0;
    virtual xt::xarray<double> forward_pass(xt::xarray<double> *X, bool training)=0;
    virtual void backward_pass(xt::xarray<double> *accum_grad, int index)=0;
    virtual void jacob_backward_pass(xt::xarray<double> *accum_grad, int index)=0;
    virtual void jacob_backward_opt_pass(xt::xarray<double> *accum_grad, int index)=0;
    virtual Shape output_shape()=0;
};
class Dense: public Layer {
    string name = "Dense";
    int n_units;
    xt::xarray<double> layer_input;
    xt::xarray<double> W, w0;
    Optimizer *W_opt, *w0_opt;
public:
    Dense(int n_units, Shape, bool first_layer, bool latent_layer);
    Dense(int n_units, Shape input_shape);
    explicit Dense(int n_units);
    string layer_name() override;
    int parameters() override;
    void initialize(string optimizer) override;
    xt::xarray<double> forward_pass(xt::xarray<double> *X, bool training) override;
    void backward_pass(xt::xarray<double> *accum_grad, int index) override;
    void jacob_backward_pass(xt::xarray<double> *accum_grad, int index) override;
    void jacob_backward_opt_pass(xt::xarray<double> *accum_grad, int index) override;
    Shape output_shape() override;
};
class Activation: public Layer {
    string name = "Activation";
    string function_name;
    ActivationFunction* activation_function;
    xt::xarray<double> layer_input;
public:
    explicit Activation(string);
    string layer_name() override;
    int parameters() override;
    void initialize(string optimizer) override;
    xt::xarray<double> forward_pass(xt::xarray<double> *X, bool training) override;
    void backward_pass(xt::xarray<double> *accum_grad, int index) override;
    void jacob_backward_pass(xt::xarray<double> *accum_grad, int index) override;
    void jacob_backward_opt_pass(xt::xarray<double> *accum_grad, int index) override;
    Shape output_shape() override;
};
class BatchNormalization: public Layer {
    string name = "BatchNormalization";
    bool initialized;
    double momentum, epsilon;
    xt::xarray<double> gamma, beta, X_centered, running_mean, running_var, stddev_inv;
    Optimizer *gamma_opt, *beta_opt;
public:
    explicit BatchNormalization(double);
    string layer_name() override;
    int parameters() override;
    void initialize(string optimizer) override;
    xt::xarray<double> forward_pass(xt::xarray<double> *X, bool training) override;
    void backward_pass(xt::xarray<double> *accum_grad, int index) override;
    void jacob_backward_pass(xt::xarray<double> *accum_grad, int index) override;
    void jacob_backward_opt_pass(xt::xarray<double> *accum_grad, int index) override;
    Shape output_shape() override;
};

#endif //ML_SPRING19_PROJECT_LAYER_H
