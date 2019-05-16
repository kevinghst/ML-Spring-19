from __future__ import print_function, division
import math
import numpy as np
import copy
from activation_functions import Sigmoid, TanH, LeakyReLU, ReLU
import time

class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def layer_name(self):
        """ The name of the layer. Used in model summary. """
        return self.__class__.__name__

    def parameters(self):
        """ The number of trainable parameters used by the layer """
        return 0

    def forward_pass(self, X, training):
        """ Propogates the signal forward in the network """
        raise NotImplementedError()

    def backward_pass(self, accum_grad, idx):
        """ Propogates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer. """
        raise NotImplementedError()

    def output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()


class Dense(Layer):
    """A fully-connected NN layer.
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    """
    def __init__(self, n_units, input_shape=None, first_layer=False, latent_layer=False):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None
        self.backprop_opt = False
        self.first_layer = first_layer
        self.latent_layer = latent_layer

    def initialize(self, optimizer):
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W  = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad, idx):
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

            # Update the layer weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def jacob_backward_pass(self, accum_grad, idx):
        start = time.time()
        W = self.W
        if idx == 1:
            accum_grad = np.einsum('ij,jk->ijk', accum_grad, W.T)
            end = time.time()
            duration = (end-start) * 1000
            print(str(idx) + ":" + str(duration))
            return accum_grad
        else:
            accum_grad = accum_grad.dot(W.T)
            end = time.time()
            duration = (end-start) * 1000
            print(str(idx) + ":" + str(duration))
            return accum_grad

    def jacob_backward_opt_pass(self, past_grad, idx):
        start = time.time()
        W = self.W
        if self.latent_layer:
            accum_grad = past_grad.dot(W.T)
            end = time.time()
            duration = (end-start) * 1000
            print(str(idx) + ":" + str(duration))
            return (past_grad, W)
        else:
            a_grad, b_grad = past_grad
            temp = b_grad.dot(W.T)
            accum_grad = np.einsum('ijk,ikp->ijp', a_grad, temp)

            if self.first_layer:
                end = time.time()
                duration = (end-start) * 1000
                print(str(idx) + ":" + str(duration))
                return accum_grad
            else:
                end = time.time()
                duration = (end-start) * 1000
                print(str(idx) + ":" + str(duration))
                return (a_grad, temp)

    def output_shape(self):
        return (self.n_units, )


activation_functions = {
    'sigmoid': Sigmoid,
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
    'relu': ReLU
}

class Activation(Layer):
    """A layer that applies an activation operation to the input.

    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, name):
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True
        self.backprop_opt = False
        self.latent_layer = False

    def layer_name(self):
        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad, idx):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def jacob_backward_pass(self,accum_grad, idx):
        start = time.time()
        act_grad = self.activation_func.gradient(self.layer_input)
        if idx == 0:
            end = time.time()
            duration = (end-start) * 1000
            print(str(idx) + ":" + str(duration))
            return act_grad
        else:
            arr = np.einsum('ijk,ik -> ijk',accum_grad, act_grad)
            end = time.time()
            duration = (end-start) * 1000
            print(str(idx) + ":" + str(duration))
            return arr

    def jacob_backward_opt_pass(self, past_grad, idx):
        start = time.time()
        a_grad, b_grad = past_grad
        act_grad = self.activation_func.gradient(self.layer_input)
        act_grad = map(np.diagflat, act_grad)
        act_grad = np.array(list(act_grad))

        if len(b_grad.shape) == 2:
            temp = np.tensordot(act_grad,b_grad.T,axes=(1,1)).swapaxes(1,2)
        else:
            temp = np.einsum('ijk,ikp->ijp', b_grad, act_grad)

        accum_grad = np.einsum('ijk,ikp->ijp', a_grad, temp)

        end = time.time()
        duration = (end-start) * 1000
        print(str(idx) + ":" + str(duration))
        return (a_grad, temp)

    def output_shape(self):
        return self.input_shape


class BatchNormalization(Layer):
    """Batch normalization.
    """
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.trainable = True
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None
        self.backprop_opt = False
        self.latent_layer = False

    def initialize(self, optimizer):
        # Initialize the parameters
        self.gamma  = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)
        # parameter optimizers
        self.gamma_opt  = copy.copy(optimizer)
        self.beta_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    def forward_pass(self, X, training=True):
        # Initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = np.mean(X, axis=0)
            self.running_var = np.var(X, axis=0)

        if training and self.trainable:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Statistics saved for backward pass
        self.X_centered = X - mean
        self.stddev_inv = 1 / np.sqrt(var + self.eps)

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta

        return output

    def backward_pass(self, accum_grad, idx):
        # Save parameters used during the forward pass
        gamma = self.gamma

        # If the layer is trainable the parameters are updated
        if self.trainable:
            X_norm = self.X_centered * self.stddev_inv
            grad_gamma = np.sum(accum_grad * X_norm, axis=0)
            grad_beta = np.sum(accum_grad, axis=0)

            self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update(self.beta, grad_beta)

        batch_size = accum_grad.shape[0]

        # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)
        accum_grad = (1 / batch_size) * gamma * self.stddev_inv * (
            batch_size * accum_grad - np.sum(accum_grad, axis=0) - self.X_centered * self.stddev_inv**2 * np.sum(accum_grad * self.X_centered, axis=0)
        )

        return accum_grad

    def jacob_backward_pass(self,accum_grad, idx):
        start = time.time()
        batch_size = accum_grad.shape[0]
        gamma = self.gamma
        expand_X_centered = np.apply_along_axis(np.tile, -1, self.X_centered, (accum_grad.shape[1],1))

        accum_grad = (1/batch_size) * gamma * self.stddev_inv * (
            batch_size * accum_grad - np.sum(accum_grad, axis=0) - expand_X_centered * self.stddev_inv**2 * np.sum(accum_grad * expand_X_centered, axis=0)
        )
        end = time.time()
        duration = (end-start) * 1000
        print(str(idx) + ":" + str(duration))
        return accum_grad

    def output_shape(self):
        return self.input_shape
