//
// Created by Robbie on 5/4/2019.
//

#include <chrono>
#include "layer.h"

#define trace(input) do { if (1) { cout << input << endl; } } while(0)
#define here() do { cout << "  here" << endl; } while (0)
#define there() do { cout << "  there" << endl; } while (0)

void Layer::set_input_shape(Shape input_shape) {
    this->input_shape = input_shape;
}
xt::xarray<double> Layer::dot(xt::xarray<double> *a, xt::xarray<double> *b) {
//    Use this definition if xtensor_blas doesn't compile
    xt::xarray<double> mult = xt::zeros<double>({a->shape(0), b->shape(1)});
    for (int i=0; i<a->shape(0); i++) {
        for (int j=0; j<b->shape(1); j++) {
            for (int k=1; k<a->shape(1); k++) {
                mult(i,j) += (*a)(i,k) * (*b)(k,j);
            }
        }
    }
//    xt::xarray<double> mult = xt::linalg::dot(*a, *b);
    return mult;
}

Dense::Dense(int n_units, Shape input_shape, bool first_layer, bool latent_layer) {
    this->n_units = n_units;
    this->input_shape = input_shape;
    this->first_layer = first_layer;
    this->latent_layer = latent_layer;
}
Dense::Dense(int n_units, Shape input_shape) {
    this->n_units = n_units;
    this->input_shape = input_shape;
    this->first_layer = false;
    this->latent_layer = false;
}
Dense::Dense(int n_units) {
    this->n_units = n_units;
    this->input_shape = Shape(1, 1);
    this->first_layer = false;
    this->latent_layer = false;
}
void Dense::initialize(string optimizer) {
    this->trainable = true;
    double limit = 1.0 / sqrt(this->input_shape.first);
    this->W = xt::random::rand({this->input_shape.first, this->n_units}, -limit, limit);
    this->w0 = xt::zeros<double>({1, this->n_units});
    if (optimizer == "adam") {
        this->W_opt = new Adam();
        this->w0_opt = new Adam();
    } else if (optimizer == "adadelta"){
        this->W_opt = new Adadelta();
        this->w0_opt = new Adadelta();
    }
}
string Dense::layer_name() {
    return this->name;
}
int Dense::parameters() {
    return this->W.shape(0) * this->W.shape(1) + this->w0.shape(0) * this->w0.shape(1);
}
xt::xarray<double> Dense::forward_pass(xt::xarray<double> *X, bool training) {
    this->layer_input = *X;
//    trace("    X (" << X->get_rows_number() << "," << X->get_columns_number() << ")");
//    trace("    W (" << W.get_rows_number() << "," << W.get_columns_number() << ")");
//    trace("    w0(" << w0.get_rows_number() << "," << w0.get_columns_number() << ")");
    return this->dot(X, &this->W) + this->w0;
}
void Dense::backward_pass(xt::xarray<double> *accum_grad, int index) {
    xt::xarray<double> W = xt::transpose(this->W);
    if (this->trainable) {
        // calc gradients w.r.t. layer weights
        xt::xarray<double> input_transp = xt::transpose(this->layer_input);
        auto grad_W = this->dot(&input_transp, accum_grad);
        xt::xarray<double> grad_w0 = xt::sum(*accum_grad, 0);
        // update layer weights
        this->W = this->W_opt->update(&this->W, &grad_W);
        this->w0 = this->w0_opt->update(&this->w0, &grad_w0);
    }
    *accum_grad = this->dot(accum_grad, &W);
}
void Dense::jacob_backward_pass(xt::xarray<double> *accum_grad, int index) {
    xt::xarray<double> W = xt::transpose(this->W);
//    auto start = chrono::system_clock::now();
    if (index == 1) {
        // accum_grad = np.einsum('ij,jk->ijk', accum_grad, W.T)
        xt::xarray<double> temp = xt::zeros<double>({accum_grad->shape(0), accum_grad->shape(1), W.shape(1)});
        for (int i=0; i<accum_grad->shape(0); i++) {
            for (int j=0; j<accum_grad->shape(1); j++) {
                for (int k=0; k<W.shape(1); k++) {
                    temp(i,j,k) = (*accum_grad)(i,j) * W(j,k);
                }
            }
        }
        *accum_grad = temp;
    } else {
        *accum_grad = this->dot(accum_grad, &W);
    }
//    auto end = chrono::system_clock::now();
//    trace("  " << index << ": " << (chrono::duration<double>(end - start)).count());
}
void Dense::jacob_backward_opt_pass(xt::xarray<double> *accum_grad, int index) {
    xt::xarray<double> W = xt::transpose(this->W);
//    auto start = chrono::system_clock::now();
    if (this->latent_layer) {
        *accum_grad = this->dot(accum_grad, &W);
    } else {
        auto a_grad = *accum_grad;
        auto b_grad = this->dot(accum_grad, &W);
        if (this->first_layer){
//            *accum_grad = einsum('ijk,ikp->ijp', a_grad, b_grad);
            xt::xarray<double> temp = xt::zeros<double>({a_grad.shape(0), a_grad.shape(1), b_grad.shape(2)});
            for (int i=0; i<a_grad.shape(0); i++) {
                for (int j=0; j<a_grad.shape(1); j++) {
                    for (int p=0; p<b_grad.shape(2); p++) {
                        for (int k=0; k<b_grad.shape(1); k++) {
                            temp(i,j,k) += a_grad(i,j,k) * b_grad(i,k,p);
                        }
                    }
                }
            }
            *accum_grad = temp;
        }
    }
//    auto end = chrono::system_clock::now();
//    trace("  " << index << ": " << (chrono::duration<double>(end - start)).count());
}

Shape Dense::output_shape() {
    return Shape (this->n_units, 1);
}


Activation::Activation(string function_name) {
    this->function_name = function_name;
    this ->trainable = true;
    this->first_layer = false;
    this->latent_layer = false;
    if (function_name == "sigmoid")
        this->activation_function = new Sigmoid();
    else if (function_name == "tanh")
        this->activation_function = new TanH();
    else if (function_name == "leaky_relu")
        this->activation_function = new LeakyReLU();
}
string Activation::layer_name() {
    return this->name + " (" + this->function_name + ")";
}
int Activation::parameters() {
    return 0;
}
xt::xarray<double> Activation::forward_pass(xt::xarray<double> *X, bool training) {
    this->layer_input = *X;
    return this->activation_function->function(X);
}

void Activation::backward_pass(xt::xarray<double> *accum_grad, int index) {
    *accum_grad = *accum_grad * this->activation_function->gradient(&layer_input);
}
void Activation::jacob_backward_pass(xt::xarray<double> *accum_grad, int index){
//    auto start = chrono::system_clock::now();

    auto act_grad = this->activation_function->gradient(&this->layer_input);
    if (index == 0) {
        *accum_grad = act_grad;
    } else {
//        *accum_grad = np.einsum('ijk,ik -> ijk', accum_grad, act_grad);
        xt::xarray<double> temp = xt::zeros<double>(accum_grad->shape());
        for (int i=0; i<accum_grad->shape(0); i++) {
            for (int j=0; j<accum_grad->shape(1); j++) {
                for (int k=0; k<accum_grad->shape(1); k++) {
                    temp(i,j,k) = (*accum_grad)(i,j,k) * act_grad(i,k);
                }
            }
        }
        *accum_grad = temp;
    }
//    auto end = chrono::system_clock::now();
//    trace("  " << index << ": " << (chrono::duration<double>(end - start)).count());
}
void Activation::jacob_backward_opt_pass(xt::xarray<double> *accum_grad, int index){
//    auto start = chrono::system_clock::now();
    auto a_grad = *accum_grad;
    auto b_grad = *accum_grad;
    auto act_grad = this->activation_function->gradient(&this->layer_input);
    act_grad = xt::diag(xt::flatten(act_grad));

    xt::xarray<double> temp;
    if (b_grad.dimension() == 2) {
//        temp = xt::linalg::tensordot(act_grad, xt::transpose(b_grad), {1}, {1});
        xt::xarray<double> transp = xt::transpose(b_grad);
        temp = this->dot(&act_grad, &transp);
    } else {
//        temp = np.einsum('ijk,ikp->ijp', b_grad, act_grad)
        temp = xt::zeros<double>({b_grad.shape(0), b_grad.shape(1), act_grad.shape(2)});
        for (int i=0; i<b_grad.shape(0); i++) {
            for (int j=0; j<b_grad.shape(1); j++) {
                for (int p=0; p<act_grad.shape(2); p++) {
                    for (int k=0; k<act_grad.shape(1); k++) {
                        temp(i,j,k) += b_grad(i,j,k) * act_grad(i,k,p);
                    }
                }
            }
        }
    }

//    accum_grad = np.einsum('ijk,ikp->ijp', a_grad, temp)
    xt::xarray<double> temp2 = xt::zeros<double>({a_grad.shape(0), a_grad.shape(1), temp.shape(2)});
    for (int i=0; i<a_grad.shape(0); i++) {
        for (int j=0; j<a_grad.shape(1); j++) {
            for (int p=0; p<temp.shape(2); p++) {
                for (int k=0; k<temp.shape(1); k++) {
                    temp2(i,j,k) += a_grad(i,j,k) * temp(i,k,p);
                }
            }
        }
    }
    *accum_grad = temp2;

//    auto end = chrono::system_clock::now();
//    trace("  " << index << ": " << (chrono::duration<double>(end - start)).count());
}
Shape Activation::output_shape() {
    return this->input_shape;
}
void Activation::initialize(string optimizer) {}


BatchNormalization::BatchNormalization(double momentum) {
    this->momentum = momentum;
    this->epsilon = 0.01;
    this->trainable = true;
    this->initialized = false;
}
void BatchNormalization::initialize(string optimizer) {
    this->gamma = xt::ones<double>({this->input_shape.first, this->input_shape.second});
    this->beta = xt::zeros<double>({this->input_shape.first, this->input_shape.second});
    if (optimizer == "adam") {
        this->gamma_opt = new Adam();
        this->beta_opt = new Adam();
    } else if (optimizer == "adadelta"){
        this->gamma_opt = new Adadelta();
        this->beta_opt = new Adadelta();
    }
}
string BatchNormalization::layer_name() {
    return this->name;
}
int BatchNormalization::parameters() {
//    return xt::prod(this->gamma.shape()) + xt::prod(this->beta.shape());
    return this->gamma.shape(0) * this->gamma.shape(1) + this->beta.shape(0) * this->beta.shape(1);
}
xt::xarray<double> BatchNormalization::forward_pass(xt::xarray<double> *X, bool training) {
    if (!this->initialized) {
        this->running_mean = xt::mean(*X, 0);
        this->running_var = xt::variance(*X, {0});
        this->initialized = true;
    }
    xt::xarray<double> mean, var;
    if (training && this->trainable) {
        mean = xt::mean(*X, 0);
        var = xt::variance(*X, {0});
        this->running_mean = this->running_mean * this->momentum + mean * (1 - this->momentum);
        this->running_var = this->running_var * this->momentum + var * (1 - this->momentum);
    } else {
        mean = this->running_mean;
        var = this->running_var;
    }
    // stats to save for backward pass
    this->X_centered = *X - mean; // element-wise xt::xarray minus row vector: each row of X gets the mean subtracted
    this->stddev_inv =1.0 / xt::sqrt(var + this->epsilon);
//    this->X_centered.print_preview();
//    var.print_unique();
//    this->stddev_inv.print_unique();

    auto X_norm = this->X_centered * this->stddev_inv;
//    X_norm.print_preview();
//    this->gamma.print_preview();
//    this->beta.print_preview();
//    trace("    X_norm (" << X_norm.get_rows_number() << "," << X_norm.get_columns_number() << ")");
//    trace("    gamma  (" << this->gamma.get_rows_number() << "," << this->gamma.get_columns_number() << ")");
//    trace("    beta   (" << this->beta.get_rows_number() << "," << this->beta.get_columns_number() << ")");
    return this->gamma * X_norm + this->beta;
}
void BatchNormalization::backward_pass(xt::xarray<double> *accum_grad, int index) {
    auto gamma = this->gamma;

    if (this->trainable) {
        auto X_norm = this->X_centered * this->stddev_inv;
        xt::xarray<double> grad_gamma = xt::sum(*accum_grad * X_norm, 0);
        xt::xarray<double> grad_beta = xt::sum(*accum_grad, 0);
//        trace("    accum_grad (" << accum_grad->get_rows_number() << "," << accum_grad->get_columns_number() << ")");
//        trace("    X_norm     (" << X_norm.get_rows_number() << "," << X_norm.get_columns_number() << ")");
//        trace("    grad_gamma (" << grad_gamma.get_rows_number() << "," << grad_gamma.get_columns_number() << ")");
//        trace("    gamma      (" << this->gamma.get_rows_number() << "," << this->gamma.get_columns_number() << ")");
        this->gamma = this->gamma_opt->update(&this->gamma, &grad_gamma);
        this->beta = this ->beta_opt->update(&this->beta, &grad_beta);
    }
    int batch_size = accum_grad->shape(0);
//    trace("    accum_grad (" << accum_grad->get_rows_number() << "," << accum_grad->get_columns_number() << ")");
//    trace("    X_centered (" << this->X_centered.get_rows_number() << "," << this->X_centered.get_columns_number() << ")");
    *accum_grad = (1.0/batch_size) * gamma * this->stddev_inv * (
            (batch_size * *accum_grad)
            - xt::sum(*accum_grad, 0)
            - (this->X_centered * xt::pow(this->stddev_inv, 2) * xt::sum(*accum_grad * this->X_centered, 0))
            );
//    trace("    accum_grad(" << accum_grad->get_rows_number() << "," << accum_grad->get_columns_number() << ")");
}
Shape BatchNormalization::output_shape() {
    return this->input_shape;
}
void BatchNormalization::jacob_backward_pass(xt::xarray<double> *accum_grad, int index) {}
void BatchNormalization::jacob_backward_opt_pass(xt::xarray<double> *accum_grad, int index) {}
