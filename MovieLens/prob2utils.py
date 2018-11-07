# Solution routine for matrix factorization
# CSCI 6140 HW4

import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    # temp = Yij - np.dot(Ui, Vj)
    temp = Yij - np.dot(Ui, Vj.T)
    return eta * (temp * Vj - reg * Ui)

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    # temp = Yij - np.dot(Vj, Ui)
    temp = Yij - np.dot(Vj, Ui.T)
    return eta * (temp * Ui - reg * Vj)


# def rmse_score(U, V, Y):
#     matrix = np.zeros((len(U), len(V)))
#     for temp in Y:
#         matrix[temp[0] - 1][temp[1] - 1] = temp[2]
#     I = matrix != 0
#     ME = I * (matrix - np.dot(U, V.T))  # Errors between real and predicted ratings
#     MSE = ME ** 2
#     return np.sqrt(np.sum(MSE) / np.sum(I))  # sum of squared errors

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    matrix = np.zeros((len(U), len(V)))
    for temp in Y:
        matrix[temp[0] - 1][temp[1] - 1] = temp[2]
    return get_err2(U, V, matrix, reg)
    # I = matrix != 0
    # ME = I * (matrix - np.dot(U, V.T))
    # e = 0.5 * (np.sum(pow(ME, 2))
    #            + reg * (pow(np.linalg.norm(U), 2) + pow(np.linalg.norm(V), 2)))
    # return e / np.prod(matrix.shape)
    # return e / np.sum(I)

def get_err2(U, V, matrix, reg=0.0):
    I = matrix != 0
    ME = I * (matrix - np.dot(U, V.T))
    e = 0.5 * (np.sum(pow(ME, 2))
               + reg * (pow(np.linalg.norm(U), 2) + pow(np.linalg.norm(V), 2)))
    return e / np.prod(matrix.shape)

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    matrix = np.zeros((M, N))
    for temp in Y:
        matrix[temp[0] - 1][temp[1] - 1] = temp[2]
    U = np.random.rand(M, K) - 0.5  # 943 * 10
    V = np.random.rand(N, K) - 0.5  # 1682 * 10
    errs = [get_err(U, V, Y, reg)]
    M_new = np.arange(M)
    N_new = np.arange(N)

    for t in range(max_epochs):
        np.random.shuffle(M_new)
        np.random.shuffle(N_new)
        for i in range(len(M_new)):
            for j in range(len(N_new)):
                if (matrix[i][j] > 0):
                    # U[i, :] += grad_U(U[i, :], matrix[i][j], V[:, j], reg, eta)
                    # V[:, j] += grad_V(V[:, j], matrix[i][j], U[i, :], reg, eta)
                    U[i, :] += grad_U(U[i, :], matrix[i][j], V[j, :], reg, eta)
                    V[j, :] += grad_V(V[j, :], matrix[i][j], U[i, :], reg, eta)
        errs.append(get_err2(U, V, matrix, reg))
        #print(errs)
        if (((errs[-2] - errs[-1]) / (errs[0] - errs[1])) <= eps):
            break
    return U, V, errs[-1]