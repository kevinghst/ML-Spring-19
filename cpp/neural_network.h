//
// Created by Robbie on 5/3/2019.
//

#ifndef ML_SPRING19_PROJECT_NEURAL_NETWORK_H
#define ML_SPRING19_PROJECT_NEURAL_NETWORK_H

#include <string>
#include <vector>
#include "loss_function.h"
#include "optimizer.h"
#include "layer.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>

using namespace std;

class NeuralNetwork {
    string optimizer;
    Loss* loss_function;
    xt::xarray<double> errors; // 0 is training, 1 is validation
    //skipping progress bar
    //skipping validation data

public:
    int batch_size = 0;
    vector<Layer*> layers;

    NeuralNetwork(string optimizer, Loss *loss_function);
//    void set_trainable(bool);
    void add(Layer*);
//    double test_on_batch(xt::xarray<double>*, xt::xarray<double>*);
    double train_on_batch(xt::xarray<double>*, xt::xarray<double>*);
//    xt::xarray<double> fit(xt::xarray<double>*, xt::xarray<double>*, int, int);
    xt::xarray<double> _forward_pass(xt::xarray<double>*, bool);
    void _backward_pass(xt::xarray<double>*);
    xt::xarray<double> _jacobian();
    xt::xarray<double> _jacobian_opt();
    void summary(string);
    xt::xarray<double> predict(xt::xarray<double> *X);
};


#endif //ML_SPRING19_PROJECT_NEURAL_NETWORK_H
