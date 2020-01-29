# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np


def svm_naive_loss_function(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    loss = 0.0
    #############################################################################
    # TODO: Calculez le gradient "dW" et la perte "loss" et stockez le résultat #
    #  dans "dW et dans "loss".                                                 #
    #  Pour cette implementation, vous devez naivement boucler sur chaque pair  #
    #  (X[i],y[i]), déterminer la perte (loss) ainsi que le gradient (voir      #
    #  exemple dans les notes de cours).  La loss ainsi que le gradient doivent #
    #  être par la suite moyennés.  Et, à la fin, n'oubliez pas d'ajouter le    #
    #  terme de régularisation L2 : reg*||w||^2                                 #
    #############################################################################
    classNum = W.shape[1]
    batchSize = X.shape[0]

    for i, x in enumerate(X):
        scores = W.T.dot(x)
        targetScore = scores[y[i]]

        for j in range(classNum):
            if j == y[i]:
                continue

            margin = scores[j] - targetScore + 1
            loss += max(0, margin)

            if margin > 0:
                dW.T[j] += x
                dW.T[y[i]] -= x

    # averaging loss and dW according to batch's size
    loss /= batchSize
    dW /= batchSize

    # add regularization
    loss += 0.5 * reg * np.linalg.norm(W)**2
    dW += reg * W
    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    return loss, dW


def svm_vectorized_loss_function(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO: Implémentez une version vectorisée de la fonction de perte SVM.     #
    # Veuillez mettre le résultat dans la variable "loss".                      #
    # NOTE : Cette fonction ne doit contenir aucune boucle                      #
    #############################################################################
    batchSize = X.shape[0]

    # compute scores and isolate targets' scores
    scores = X.dot(W)
    targetScores = scores[range(batchSize), y].reshape((-1,1)) # transpose to column vector

    # compute margins
    margins = np.maximum(0, scores - targetScores + 1)
    margins[range(batchSize), y] = 0
    loss = np.sum(margins)

    # averaging loss according to batch's size
    loss /= batchSize

    # add regularisation
    loss += 0.5 * reg * np.linalg.norm(W)**2

    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    #############################################################################
    # TODO: Implémentez une version vectorisée du calcul du gradient de la      #
    #  perte SVM.                                                               #
    # Stockez le résultat dans "dW".                                            #
    #                                                                           #
    # Indice: Au lieu de calculer le gradient à partir de zéro, il peut être    #
    # plus facile de réutiliser certaines des valeurs intermédiaires que vous   #
    # avez utilisées pour calculer la perte.                                    #
    #############################################################################

    # create mask for X with positive margins
    marginsMask = (margins > 0).astype(int)

    # set the data's target to count the number of these examples where margin > 0
    marginsMask[range(batchSize), y] = - marginsMask.sum(axis=1)

    # compute gradient
    dW = X.T.dot(marginsMask)

    # averaging gradient according to batch's size
    dW /= batchSize

    # add regularisation
    dW += reg * W

    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    return loss, dW
