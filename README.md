# Optimizing Jacobian Computation for Autoencoders

This repo contains Python and C++ implementations for the standard and optimized versions of the algorithm used to compute the Jacobians of an autoencoder.

The methods **_jacobian()** and **_jacobian_opt()** in *neural_network* are the standard and optimized backpropagation algorithms respectively. The methods **jacob_backward_pass()** and **jacob_backward_opt_pass()** from *layers* 
are used to propagate the Jacobian across each layer object.

The underlying autoencoder infrastructure is based on Erik Lindernoren's *ML-From-Scratch* project.

