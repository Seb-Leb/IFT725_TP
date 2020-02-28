import numpy as np

from ift725.layers import *
from ift725.quick_layers import *
from ift725.layer_combo import *


class CustomConvolutionalNet(object):
    """
    A custom convolutional network with the following architecture:

    ([conv-relu-bn]xN1-[pool])xN2 - [FC-relu]xM - [softmax]

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-2, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
         of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ####################
        #  INITIALIZATION  #
        ####################

        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)

        self.params['W2'] = weight_scale * np.random.randn(8192, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        #############################
        # END OF THE INITIALIZATION #
        #############################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):

        # pass conv_param to the forward pass for the convolutional layer
        filter_size_1 = self.params['W1'].shape[2]
        conv_param_1 = {'stride': 1, 'pad': (filter_size_1 - 1) / 2}

        filter_size_2 = 5
        conv_param_2 = {'stride': 1, 'pad': (filter_size_1 - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # pass gamma, beta, bn_param to the forward pass for the forward_spatial_batch_normalization layer
        gamma_1 = np.random.randn(32)
        beta_1 = np.random.randn(32)
        bn_param = {'mode': 'train'}

        scores = None

        #########################
        #  FORWARD PROPAGATION  #
        #########################

        # Forward Pass: ([conv-relu-bn]xN1-[pool])xN2 - [FC-relu]xM - FC - [softmax]

        # N2 = 1
        # [conv-relu-bn]*(N1=1)
        out1, cache_conv = forward_convolutional_relu(X, self.params['W1'], self.params['b1'], conv_param_1)

        out2, cache_bn = forward_spatial_batch_normalization(out1, gamma_1, beta_1, bn_param)

        # [pool]
        output_pooling, cache_pooling = max_pool_forward_fast(out2, pool_param)

        # print("output_pooling.shape", output_pooling.shape)

        # M
        out2, cache_fc_relu = forward_fully_connected_transform_relu(output_pooling, self.params['W2'], self.params['b2'])

        out3, cache_fc = forward_fully_connected(out2, self.params['W3'], self.params['b3'])

        scores = out3

        loss, grads = 0, {}

        loss, dout = softmax_loss(out3, y)

        loss += self.reg * np.sum([np.sum(self.params['W%d' % i] ** 2) for i in [1, 2, 3]])

        ##################################
        # END OF THE FORWARD PROPAGATION #
        ##################################

        if y is None:
            return scores

        ########################
        # BACKWARD PROPAGATION #
        ########################

        # Backward Pass: ([conv-relu-bn]xN1-[pool])xN2 - [FC-relu]xM - FC -[softmax]

        dout, grads['W3'], grads['b3'] = backward_fully_connected(dout, cache_fc)
        grads['W3'] += 2 * self.reg * self.params['W3']

        dout, grads['W2'], grads['b2'] = backward_fully_connected_transform_relu(dout, cache_fc_relu)
        grads['W2'] += 2 * self.reg * self.params['W2']

        dout = max_pool_backward_fast(dout, cache_pooling)

        dout, dgamma, dbeta = backward_spatial_batch_normalization(dout, cache_bn)

        _, grads['W1'], grads['b1'] = backward_convolutional_relu(dout, cache_conv)
        grads['W1'] += 2 * self.reg * self.params['W1']

        ###################################
        # END OF THE BACKWARD PROPAGATION #
        ###################################

        return loss, grads