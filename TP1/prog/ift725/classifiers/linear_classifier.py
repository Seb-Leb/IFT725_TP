# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np
from ift725.classifiers.linear_svm import *
from ift725.classifiers.softmax import *


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Échantillonnez "batch_size" éléments à partir des données       #
            #  d'apprentissage et des étiquettes associées à utiliser dans cette    #
            #  étape de descente de gradient.                                       #
            # Stockez les données dans "X_batch" et les étiquettes correspondantes  #
            # dans "y_batch"; après l'échantillonnage, "X_batch" devrait être de la #
            # forme (batch_size, dim) et "y_batch" devrait avoir la                 #
            # forme (batch_size,).                                                  #
            #                                                                       #
            # Indice: Utilisez np.random.choice pour générer les indices.           #
            # L'échantillonnage avec remplacement est plus rapide que               #
            # l'échantillonnage sans remplacement.                                  #
            #########################################################################

            #########################################################################
            #                      FIN DE VOTRE CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO: Mise à jour des poids en utilisant le gradient et la vitesse    #
            #  d'apprentissagethe weights using the gradient and the learning rate. #
            #########################################################################

            #########################################################################
            #                      FIN DE VOTRE CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: (N, D) array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        labels_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO: Implémentez cette fonction.                                       #
        # Stockez les étiquettes prédites dans "labels_pred".                     #
        ###########################################################################

        ###########################################################################
        #                          FIN DE VOTRE CODE                              #
        ###########################################################################
        return labels_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_vectorized_loss_function(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_vectorized_loss_function(self.W, X_batch, y_batch, reg)
