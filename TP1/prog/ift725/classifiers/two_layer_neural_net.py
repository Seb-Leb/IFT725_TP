# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np


class TwoLayerNeuralNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {
            'W1': std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        Weights1, biases1 = self.params['W1'], self.params['b1']
        Weights2, biases2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        H, C = Weights2.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Effectuez la propoagation avant en calculant le scores des classes  #
        #  pour chaque élément de la mini-batch X.  NOTE: cette opération ne        #
        #  requière pas de softmax, ce dernier étant calculé plus bas.              #
        #  NOTE : votre code ne doit contenir aucune boucle for                     #
        # Stocker le résultat dans la variable "scores", qui devrait être un        #
        # tableau de la forme (N, C).                                               #
        #############################################################################

        #scores = ...
        #############################################################################
        #                             FIN DE VOTRE CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Terminez la propagation avant et calculez la softmax +              #
        #  l'entropie-croisée.                                                      #
        # Cela devrait inclure à la fois la perte de données et la régularisation   #
        # L2 pour les poids Weights1, Weights2, biais 1 et bias 2. Stockez le       #
        # résultat dans la variable "loss", qui doit être une valeur scalaire.      #
        # NOTE : votre code doit être linéarisé et donc ne contenir AUCUNE boucle   #
        #############################################################################
        loss = loss*0

        #############################################################################
        #                             FIN DE VOTRE CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Calculez la passe rétrograde (backward pass), en calculant les      #
        #  dérivées des poids et des biais.                                         #
        # Stockez les résultats dans le dictionnaire "grads". Par exemple,          #
        # "grads['W1']" devrait emmagasiner le gradient de W1, et être une matrice  #
        # de la même taille.                                                        #
        #############################################################################
        # Nomenclature:
        #   L   loss relative to all samples
        #   Li  loss relative to each sample
        #   f   class scores (N, C)
        #   f1  pre-activation of the 1st layer (N, H)
        #   a1  activation of the 1st layer (N, H)


        grads['W1'] = 0
        grads['W2'] = 0
        grads['b1'] = 0
        grads['b2'] = 0
        #############################################################################
        #                             FIN DE VOTRE CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.98,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            data_batch = None
            labels_batch = None

            #########################################################################
            # TODO: Créer une minibatch aléatoire de données et d'étiquettes        #
            #  d'entrainement.                                                      #
            # Stockez-les dans "data_batch" and "labels_batch" respectivement.      #
            #########################################################################

            #########################################################################
            #                            FIN DE VOTRE CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X=data_batch, y=labels_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Utilisez les gradients du dictionnaire grads pour mettre à jour #
            #  les paramètres du réseau de neuronnes (stokés dans le dictionnaire   #
            #  "self.params") en utilisant une descente de gradient stochastique.   #
            # Vous aurez besoin d'utiliser les gradients stockés dans le            #
            # dictionnaire "grads" défini précédemment.                             #
            #########################################################################


            #########################################################################
            #                            FIN DE VOTRE CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(data_batch) == labels_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implémentez cette fonction; elle devrait être TRÈS simple!        #
        # Indice : vous pouvez appeler des fonctions déjà codées...               #
        ###########################################################################


        ###########################################################################
        #                             FIN DE VOTRE CODE                           #
        ###########################################################################

        return y_pred
