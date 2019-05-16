//
// Created by Robbie on 5/3/2019.
//
#include <iostream>
#include <iomanip>
#include <fstream>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xview.hpp>
#include "optimizer.h"
#include "loss_function.h"
#include "layer.h"
#include "neural_network.h"

using namespace std;
using namespace xt::placeholders;

#define trace(input) do { if (1) { cout << input << endl; } } while(0)
#define here() do { cout << "  here" << endl; } while (0)
#define there() do { cout << "  there" << endl; } while (0)

class Autoencoder {
    int img_rows = 28;
    int img_cols = 28;
    int img_dim = img_rows * img_cols;
    int latent_dim = 16;

    string optimizer;
    Loss *loss_function;
    NeuralNetwork* autoencoder;

    xt::xarray<double> X, testX;

    NeuralNetwork build_encoder(string optimizer, Loss *loss_function) {
        NeuralNetwork encoder(optimizer, loss_function);
        auto *dense1 = new Dense(512, Shape(this->img_dim, 1), true, false);
        auto *act1 = new Activation("leaky_relu");
//        auto *batch1 = new BatchNormalization(0.8);
        auto *dense2 = new Dense(256);
        auto *act2 = new Activation("leaky_relu");
//        auto *batch2 = new BatchNormalization(0.8);
        auto *dense3 = new Dense(this->latent_dim, Shape(this->img_dim, 1), false, true);
        encoder.add(dense1);
        encoder.add(act1);
//        encoder->add(batch1);
        encoder.add(dense2);
        encoder.add(act2);
//        encoder->add(batch2);
        encoder.add(dense3);
        return encoder;
    }
    NeuralNetwork build_decoder(string optimizer, Loss *loss_function) {
        NeuralNetwork decoder(optimizer, loss_function);
        auto *dense1 = new Dense(256, Shape(this->latent_dim, 1));
        auto *act1 = new Activation("leaky_relu");
//        auto * batch1 = new BatchNormalization(0.8);
        auto *dense2 = new Dense(512);
        auto *act2 = new Activation("leaky_relu");
//        auto *batch2 = new BatchNormalization(0.8);
        auto *dense3 = new Dense(this->img_dim);
        auto *act3 = new Activation("tanh");
        decoder.add(dense1);
        decoder.add(act1);
//        decoder.add(batch1);
        decoder.add(dense2);
        decoder.add(act2);
//        decoder.add(batch2);
        decoder.add(dense3);
        decoder.add(act3);
        return decoder;
    }
    void save_images(int epoch) {
        int rows = 5;
        auto index = xt::random::randint<int>({rows*rows,1}, 0, X.shape(0));
        xt::xarray<double> images = xt::view(X, xt::keep(index), xt::all());
        xt::xarray<double> gen_images = this->autoencoder->predict(&images);
//        gen_images.reshape({-1, this->img_rows, this->img_cols});
        gen_images = (gen_images * -127.5) + 127.5;
//        trace(gen_images);

        string file_name = "../image_predictions/ae_" + to_string(epoch) + ".pgm";
        ofstream image_file(file_name, ofstream::out | ofstream::trunc);
        if (image_file.is_open()) {
            image_file << "P2\r\n";
            image_file << this->img_rows << " " << this->img_cols << "\r\n";
            image_file << "255\r\n";
            xt::xarray<double> image = xt::view(gen_images, xt::keep(0), xt::all());
            image.reshape({this->img_rows, this->img_cols});
//            for (int x=0; x<rows*rows)
            for (int i = 0; i< this->img_rows-1; i++) {
                for (int j = 0; j<this->img_cols-1; j++) {
                    image_file << image(i,j) << " ";
                }
                image_file << "\r\n";
            }
            image_file.close();
        }


    }
public:
    Autoencoder(xt::xarray<double> X, xt::xarray<double> testX) {
        this->X = X;
        this->testX = testX;
        this->loss_function = new SquareLoss();
        this->optimizer = "adam";
        NeuralNetwork encoder = this->build_encoder(this->optimizer, this->loss_function);
        NeuralNetwork decoder = this->build_decoder(this->optimizer, this->loss_function);
        autoencoder = new NeuralNetwork(this->optimizer, this->loss_function);
        autoencoder->layers.insert(autoencoder->layers.end(), encoder.layers.begin(), encoder.layers.end());
        autoencoder->layers.insert(autoencoder->layers.end(), decoder.layers.begin(), decoder.layers.end());
//        autoencoder->output_dim = this->img_dim;
        autoencoder->summary("Variational Autoencoder");
    }

    void train(int n_epochs, int batch_size, int save_interval) {
        this->autoencoder->batch_size = batch_size;
        this->X = (this->X - 127.5) / 127.5;
        this->testX = (this->testX - 127.5) / 127.5;
        for (int epoch = 0; epoch<n_epochs; epoch++) {
            auto index = xt::random::randint<int>({batch_size,1}, 0, X.shape(0));
            xt::xarray<double> random_batch = xt::view(X, xt::keep(index), xt::all());
            auto loss = this->autoencoder->train_on_batch(&random_batch, &random_batch);
            cout << epoch << " [D loss: " << setprecision(6) << loss << "]"<<endl;
            if (epoch % save_interval == 0)
                this->save_images(epoch);
        }
    }

};


int main (int argc, char **argv) {
    trace("Loading MNIST data..."<<endl);
    xt::xarray<double> train_data, test_data;
    ifstream train_file("../mnist_train.csv");
    if (train_file.is_open()) {
        train_data = xt::load_csv<double>(train_file);
        train_file.close();
    }
    ifstream test_file("../mnist_test.csv");
    if (test_file.is_open()) {
        test_data = xt::load_csv<double>(test_file);
        test_file.close();
    }
    xt::xarray<double> X = xt::view(train_data, xt::all(), xt::range(1,_));
    xt::xarray<double> testX = xt::view(test_data, xt::all(), xt::range(1,_));
//    trace("(" << X.shape(0) << "," << X.shape(1) << ")");
//    trace("(" << testX.shape(0) << "," << testX.shape(1) << ")");

    Autoencoder ae(X, testX);
    ae.train(2000, 50, 40);
}