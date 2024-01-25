import pandas as pd
import numpy as np
import cvxopt as cvx
from math import sqrt

from . import helpers as hlp

#####################################################
#                                                   #
#           #######     ######      ######          #
#           #     #     #     #     #               #
#           #   ###     #     #     #               #
#           ####        ######      #####           #
#           #   #       #     #     #               #
#           #    #      #     #     #               #
#           #     #     ######      #               #
#                   Neural Network                  #
#####################################################

# Inputs a DataFrame X with training data and y with class labels for training data, as well as a float value for spread
# Returns a radial basis function NN 'model' dict in the form {'W_hat': W_hat, 'W': W, 'spread': spread, 'error': error}
def rbf_NN_train(X: pd.DataFrame, y: pd.Series, spread: float = 0.5) -> dict:
    n = X.shape[0]

    H = np.zeros(shape=(n,n))

    for i in range(n):
        for j in range(n):
            w = X[j,:]
            x0 = X[i,:]
            H[i,j] = np.exp( -((np.linalg.norm(x0 - w))**2) / (2*(spread**2)) )

    W_hat = (np.linalg.pinv(H.T @ H)) @ H.T @ y
    W = X

    yt = H @ W_hat
    ypred = np.ones(shape=y.shape)
    ypred[yt < 0] = -1
    error = 1 - len(ypred[ypred == y])/y.shape[0]

    model = {'W_hat': W_hat, 'W': W, 'spread': spread, 'err': error}

    return model


# Inputs model in form: {'W_hat': W_hat, 'W': W, 'spread': spread, 'error': error}
# Outputs dict in form: {'y': y, 'ypred': ypred}
# 'y' is list of values output from final activation function
# 'ypred' is same values with 1 where y > 0 and -1 where y < 0
def rbf_NN_1v1_classify_NoBias(X: pd.DataFrame, model: dict) -> dict:
    
    W_hat = model['W_hat']
    W = model['W']
    spread = model['spread']

    n1 = X.shape[0]
    n2 = W.shape[0]

    H = np.zeros(shape=(n1, n2))
    for j in range(n2):
        for i in range(n1):
            W0 = W[j,:]
            x0 = X[i,:]
            H[i,j] = np.exp( -((np.linalg.norm(x0 - W0))**2) / (2*(spread**2)) )
    
    y = H @ W_hat
    
    ypred = np.ones(shape=(y.shape))
    ypred[y < 0] = -1

    return {'y': y, 'ypred': ypred}

def rbf_NN_train_classify(train_X: np.ndarray, train_y: np.ndarray, test_X, spread: float = 0.5) -> dict:
    model = rbf_NN_train(X=train_X, y=train_y, spread=spread)
    predictions = rbf_NN_1v1_classify_NoBias(X=test_X, model=model)
    return predictions['ypred']

############ END RBF NN ############


#####################################################
#                                                   #
#            ######    #       #    #     #         #
#           #           #     #     ##   ##         #
#           #           #     #     # # # #         #
#            #####       #   #      #  #  #         #
#                 #      #   #      #     #         #
#                 #       # #       #     #         #
#            #####         #        #     #         #
#               Support Vector Machine              #
#####################################################


# Computes the kernel function for two data points
def kernel(x1, x2, *, kernel_type='linear', arg=2):
    if kernel_type == 'linear':
        return np.dot(x1, x2.T)
    elif kernel_type == 'rbf':
        return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * arg**2))
    elif kernel_type == 'poly':
        return (np.dot(x1, x2.T) + 1)**arg
    else:
        raise ValueError('Unknown kernel type')

# Computes the kernel matrix for two sets of data points
def SVM_kernel_matrix(X1, X2, *, kernel_type='rbf', arg=0.15):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel(X1[i, :], X2[j, :], kernel_type=kernel_type, arg=arg)
    return K

# Trains a support vector machine on the data X, y using a given kernel type
def SVM_train(X, y, *, C=3.0, kernel_type='rbf', arg=0.15):
    n = X.shape[0]
    K = SVM_kernel_matrix(X, X, kernel_type=kernel_type, arg=arg)
    H = np.outer(y, y) * K
    H = H + (1e-12 * np.eye(n))
    H = H.astype('double')

    # Credit to Amir Saeed for the following code:
    cvx.solvers.options['show_progress'] = False

    P = cvx.matrix(H, tc='d')
    q = cvx.matrix(-np.ones(n), tc='d')
    G = cvx.matrix(np.vstack([-np.eye(n), np.eye(n)]), tc='d')
    h = cvx.matrix(np.hstack([np.zeros(n), np.ones(n) * C]), tc='d')
    A = cvx.matrix(y.reshape(1,-1).astype('double'), tc='d')
    b = cvx.matrix(np.zeros((1, 1)), tc='d')

    sol = cvx.solvers.qp(P, q, G, h, A, b)
    # End Amir code

    alpha = np.array(sol['x']).flatten()
    s_vector_idx = np.where(alpha > 1e-5)[0]

    eps = 1e-5
    bound_idx = np.where((alpha > eps) & (alpha < C - eps))[0]

    if len(bound_idx) > 0:
        b = sum(y[bound_idx] - H[np.ix_(bound_idx,s_vector_idx)] @ alpha[s_vector_idx] * y[bound_idx]) / len(bound_idx)
    else:
        b = 0
    
    prediction = K @ alpha + b

    tmp = np.sign(prediction)
    err = np.sum(tmp != y) / n

    alpha = (alpha * y)[s_vector_idx]

    model = {'alpha': alpha, 'b': b, 'kernel_type': kernel_type, 'arg': arg, 'C': C, 
             'sv_X': X[s_vector_idx], 'sv_y': y[s_vector_idx], 'sv_idx': s_vector_idx, 
             'err': err, 'num_svs': len(s_vector_idx)}

    return model

# Inputs a trained SVM model and a test set, and returns the predicted class
# 'y' is a magnitude, for use in the pipeline. 'ypred' is the predicted class in {-1,1}.
def SVM_1v1_classify(X, model: dict):
    alpha = model['alpha']
    b = model['b']
    kernel_type = model['kernel_type']
    arg = model['arg']
    sv_X = model['sv_X']

    K = SVM_kernel_matrix(X, sv_X, kernel_type=kernel_type, arg=arg)
    y = K @ alpha + b
    ypred = np.sign(y)

    return {'y': y, 'ypred': ypred}

############## END SVM ##############


################################
#        Parzen Window         #
################################

# Gaussian RBF kernel function
def gaussian_rbf_kernel(x0: np.ndarray, x_i: pd.DataFrame, h: float, dim: int)-> float:

    normalizer = 1 / ( ((sqrt(2*np.pi))**dim) * (h**dim) )

    norm = np.linalg.norm(x0 - x_i)
    exp = np.exp( -((norm)**2) / (2*(h**2)) )

    return (normalizer*exp)

# Takes single data point x0 and compares to all data points in dataframe and returns a single probability value
def kernel_point_probability(x0: np.ndarray, data: np.ndarray, h: float = None, dim: int = None, kernel: str = 'rbf')-> float:

    N = data.shape[0]

    if dim == None:
        if len(data.shape) == 2:
            dim = data.shape[1]
        else:
            dim = 1
    
    # This default value does not work well for me, but I'm not sure how to pick a good default.
    if h == None:
        h = 1 / sqrt(N)
        # h = 1 / N

    prob = 0.0
    if kernel == 'rbf':
        for x_i in data:
            prob += gaussian_rbf_kernel(x0=x0, x_i=x_i, h=h, dim=dim)
    else:
        raise ValueError("Kernel not defined")

    return prob / N

# Gets probability values for a point over each class in the passed-in "values"(classes) array, then returns
# the name of the class that has the highest probability value.
def kernel_point_classifier(x0: np.ndarray, data: np.ndarray, values: np.ndarray, h: float = None, dim: int = None, kernel: str = 'rbf')-> str:

    # Because I keep screwing this up:
    if type(x0) != np.ndarray:
        x0 = np.array(x0)
    if type(data) != np.ndarray:
        data = np.array(data)
    if type(values) != np.ndarray:
        values = np.array(values)

    classes = np.unique(values)
    probs = {} # max(stats, key=stats.get)

    for cls in classes:
        clsidx = np.where(values == cls)
        probs[cls] = kernel_point_probability(x0=x0, data=data[clsidx], h=h, dim=dim, kernel=kernel)
    
    # print(probs)
    return max(probs, key=probs.get)

# A full classifier that tests a set of values (test_x) over each class in a set of training data (train_x, train_y)
# and returns the predicted class for each point in the text_x array.
def parzen_kernel_classifier(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *,
                      h: float = None, dim: int = None, kernel: str = 'rbf') -> np.ndarray:
    
    # Because I keep screwing this up:
    if type(train_x) != np.ndarray:
        train_x = np.array(train_x)
    if type(train_y) != np.ndarray:
        train_y = np.array(train_y)
    if type(test_x) != np.ndarray:
        test_x = np.array(test_x)

    predicted = []
    for point in test_x:
        predicted.append(kernel_point_classifier(x0=point, data=train_x, values=train_y, h=h, dim=dim, kernel=kernel))

    return np.array(predicted)
    
############# END Parzen Window #############



################################
#       Bayes Classifier       #
################################

# A simple Bayes classifier that trains a gaussian mix model on the training data
# and predicts the class of each point in the test data using the model.  Steps:
# 1. Train model using gaussian_mix_model
# 2. Use bayes_classifier with test data and model to predict class

def gaussian_mix_model(X: np.ndarray, y: np.ndarray, is_bayes: bool = True):

    model = {}

    for class_ in np.unique(y):
        tmp = X[y == class_]

        model[class_] = {'mean': tmp.mean(axis=0), 'cov': np.cov(tmp, rowvar=False)}

    if is_bayes:
        for class_ in np.unique(y):
            model[class_]['prior'] = len(y[y == class_]) / len(y)

    return model

# Helper function for bayes_classifier
def mv_gaussian_distribution(X, mean, cov):

    n, dim = X.shape

    y = np.zeros(n)
    dist = np.zeros(n)

    for i in range(n):
        dist[i] = hlp.m_distance(X[i,:], cov, mean)**2
    y += np.exp(-0.5 * dist) / sqrt((2 * np.pi)**(dim) * np.linalg.det(cov))
    return y

def bayes_classifier(model: dict, test_X: np.ndarray):
    n, dim = test_X.shape
    classes = list(model.keys())
    num_classes = len(classes)

    y_post = np.zeros((n, num_classes))

    for i in range(num_classes):
        y_update = model[classes[i]]['prior'] * mv_gaussian_distribution(test_X, model[classes[i]]['mean'], model[classes[i]]['cov'])
        y_post[:, i] = y_update * model[classes[i]]['prior']
    
    predicted = np.argmax(y_post, axis=1)
    predicted = [classes[x] for x in predicted]

    return np.array(predicted)


############# END Bayes Classifier #############